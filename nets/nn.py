import math

import torch


def initialize_weights(model):
    for m in model.modules():
        if isinstance(m, torch.nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            m.weight.data.normal_(0, math.sqrt(2.0 / (fan_out // m.groups)))
            if m.bias is not None:
                m.bias.data.zero_()
        elif isinstance(m, torch.nn.BatchNorm2d):
            m.weight.data.fill_(1.0)
            m.bias.data.zero_()
        elif isinstance(m, torch.nn.Linear):
            m.weight.data.uniform_(-1.0 / math.sqrt(m.weight.size()[0]), 1.0 / math.sqrt(m.weight.size()[0]))
            m.bias.data.zero_()


def fuse_conv(conv, norm):
    """
    [https://nenadmarkus.com/p/fusing-batchnorm-and-conv/]
    """
    fused_conv = torch.nn.Conv2d(conv.in_channels,
                                 conv.out_channels,
                                 conv.kernel_size,
                                 conv.stride,
                                 conv.padding,
                                 groups=conv.groups, bias=True).requires_grad_(False).to(conv.weight.device)

    w_conv = conv.weight.clone().view(conv.out_channels, -1)
    w_norm = torch.diag(norm.weight.div(torch.sqrt(norm.eps + norm.running_var)))
    fused_conv.weight.copy_(torch.mm(w_norm, w_conv).view(fused_conv.weight.size()))

    b_conv = torch.zeros(conv.weight.size(0), device=conv.weight.device) if conv.bias is None else conv.bias
    b_norm = norm.bias - norm.weight.mul(norm.running_mean).div(torch.sqrt(norm.running_var + norm.eps))
    fused_conv.bias.copy_(torch.mm(w_norm, b_conv.reshape(-1, 1)).reshape(-1) + b_norm)

    return fused_conv


class Conv(torch.nn.Module):
    def __init__(self, in_ch, out_ch, activation, k=1, s=1, g=1):
        super().__init__()
        self.conv = torch.nn.Conv2d(in_ch, out_ch, k, s, k // 2, 1, g, bias=False)
        self.norm = torch.nn.BatchNorm2d(out_ch, 0.001, 0.01)
        self.relu = activation

    def forward(self, x):
        return self.relu(self.norm(self.conv(x)))

    def fuse_forward(self, x):
        return self.relu(self.conv(x))


class Residual(torch.nn.Module):
    """
    [https://arxiv.org/pdf/1801.04381.pdf]
    """

    def __init__(self, in_ch, out_ch, s, r):
        super().__init__()
        identity = torch.nn.Identity()
        relu6_fn = torch.nn.ReLU6(True)
        features = [Conv(in_ch, r * in_ch, relu6_fn) if r != 1 else identity,
                    Conv(r * in_ch, r * in_ch, relu6_fn, 3, s, r * in_ch),
                    Conv(r * in_ch, out_ch, identity)]

        self.add = s == 1 and in_ch == out_ch
        self.res = torch.nn.Sequential(*features)

    def forward(self, x):
        return self.res(x) + x if self.add else self.res(x)


class MobileNetV2(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        relu_fn = torch.nn.ReLU6(inplace=True)
        filters = [32, 16, 24, 32, 64, 96, 160, 320, 1280]
        feature = [Conv(3, filters[0], relu_fn, 3, 2),
                   Residual(filters[0], filters[1], 1, 1),
                   Residual(filters[1], filters[2], 2, 6),
                   Residual(filters[2], filters[2], 1, 6),
                   Residual(filters[2], filters[3], 2, 6),
                   Residual(filters[3], filters[3], 1, 6),
                   Residual(filters[3], filters[3], 1, 6),
                   Residual(filters[3], filters[4], 2, 6),
                   Residual(filters[4], filters[4], 1, 6),
                   Residual(filters[4], filters[4], 1, 6),
                   Residual(filters[4], filters[4], 1, 6),
                   Residual(filters[4], filters[5], 1, 6),
                   Residual(filters[5], filters[5], 1, 6),
                   Residual(filters[5], filters[5], 1, 6),
                   Residual(filters[5], filters[6], 2, 6),
                   Residual(filters[6], filters[6], 1, 6),
                   Residual(filters[6], filters[6], 1, 6),
                   Residual(filters[6], filters[7], 1, 6),
                   Conv(filters[7], filters[8], relu_fn)]

        self.feature = torch.nn.Sequential(*feature)
        self.fc = torch.nn.Sequential(torch.nn.Dropout(0.2),
                                      torch.nn.Linear(filters[8], 1000))

        initialize_weights(self)

    def forward(self, x):
        return self.fc(self.feature(x).mean((2, 3)))

    def fuse(self):
        for m in self.modules():
            if type(m) is Conv and hasattr(m, 'norm'):
                m.conv = fuse_conv(m.conv, m.norm)
                delattr(m, 'norm')
                m.forward = m.fuse_forward

        return self
