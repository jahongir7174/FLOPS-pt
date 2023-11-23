import math

import torch


def fuse_conv(conv, norm):
    """
    [https://nenadmarkus.com/p/fusing-batchnorm-and-conv/]
    """
    fused_conv = torch.nn.Conv2d(conv.in_channels,
                                 conv.out_channels,
                                 conv.kernel_size,
                                 conv.stride,
                                 conv.padding,
                                 conv.dilation,
                                 conv.groups, True).requires_grad_(False).to(conv.weight.device)

    w_conv = conv.weight.clone().view(conv.out_channels, -1)
    w_norm = torch.diag(norm.weight.div(torch.sqrt(norm.eps + norm.running_var)))
    fused_conv.weight.copy_(torch.mm(w_norm, w_conv).view(fused_conv.weight.size()))

    b_conv = torch.zeros(conv.weight.size(0), device=conv.weight.device) if conv.bias is None else conv.bias
    b_norm = norm.bias - norm.weight.mul(norm.running_mean).div(torch.sqrt(norm.running_var + norm.eps))
    fused_conv.bias.copy_(torch.mm(w_norm, b_conv.reshape(-1, 1)).reshape(-1) + b_norm)

    return fused_conv


class Conv(torch.nn.Module):
    def __init__(self, in_ch, out_ch, activation, k=1, s=1, p=0, d=1, g=1):
        super().__init__()
        self.conv = torch.nn.Conv2d(in_ch, out_ch, k, s, p, d, g, False)
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
        self.add = s == 1 and in_ch == out_ch
        self.res = torch.nn.Sequential(Conv(in_ch, r * in_ch, torch.nn.ReLU6()) if r != 1 else identity,
                                       Conv(r * in_ch, r * in_ch, torch.nn.ReLU6(), 3, s, 1, 1, r * in_ch),
                                       Conv(r * in_ch, out_ch, identity))

    def forward(self, x):
        return x + self.res(x) if self.add else self.res(x)


class MobileNetV2(torch.nn.Module):
    def __init__(self, num_classes: int = 1000):
        super().__init__()
        self.p1 = []
        self.p2 = []
        self.p3 = []
        self.p4 = []
        self.p5 = []
        filters = [3, 32, 16, 24, 32, 64, 96, 160, 320, 1280]

        # p1/2
        self.p1.append(Conv(filters[0], filters[1], torch.nn.ReLU6(), 3, 2, 1))
        self.p1.append(Residual(filters[1], filters[2], 1, 1))
        # p2/4
        self.p2.append(Residual(filters[2], filters[3], 2, 6))
        self.p2.append(Residual(filters[3], filters[3], 1, 6))
        # p3/8
        self.p3.append(Residual(filters[3], filters[4], 2, 6))
        self.p3.append(Residual(filters[4], filters[4], 1, 6))
        self.p3.append(Residual(filters[4], filters[4], 1, 6))
        # p4/16
        self.p4.append(Residual(filters[4], filters[5], 2, 6))
        self.p4.append(Residual(filters[5], filters[5], 1, 6))
        self.p4.append(Residual(filters[5], filters[5], 1, 6))
        self.p4.append(Residual(filters[5], filters[5], 1, 6))
        self.p4.append(Residual(filters[5], filters[6], 1, 6))
        self.p4.append(Residual(filters[6], filters[6], 1, 6))
        self.p4.append(Residual(filters[6], filters[6], 1, 6))
        # p5/32
        self.p5.append(Residual(filters[6], filters[7], 2, 6))
        self.p5.append(Residual(filters[7], filters[7], 1, 6))
        self.p5.append(Residual(filters[7], filters[7], 1, 6))
        self.p5.append(Residual(filters[7], filters[8], 1, 6))

        self.p1 = torch.nn.Sequential(*self.p1)
        self.p2 = torch.nn.Sequential(*self.p2)
        self.p3 = torch.nn.Sequential(*self.p3)
        self.p4 = torch.nn.Sequential(*self.p4)
        self.p5 = torch.nn.Sequential(*self.p5)

        self.fc = torch.nn.Sequential(Conv(filters[8], filters[9], torch.nn.ReLU6()),
                                      torch.nn.AdaptiveAvgPool2d(1),
                                      torch.nn.Flatten(),
                                      torch.nn.Dropout(0.2),
                                      torch.nn.Linear(filters[9], num_classes))

        # initialize weights
        for m in self.modules():
            if isinstance(m, torch.nn.Conv2d):
                fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                fan_out = fan_out // m.groups
                torch.nn.init.normal_(m.weight, 0, math.sqrt(2.0 / fan_out))
                if m.bias is not None:
                    torch.nn.init.zeros_(m.bias)
            if isinstance(m, torch.nn.Linear):
                init_range = 1.0 / math.sqrt(m.weight.size()[0])
                torch.nn.init.uniform_(m.weight, -init_range, init_range)
                if m.bias is not None:
                    torch.nn.init.zeros_(m.bias)

    def forward(self, x):
        x = self.p1(x)
        x = self.p2(x)
        x = self.p3(x)
        x = self.p4(x)
        x = self.p5(x)

        return self.fc(x)

    def fuse(self):
        for m in self.modules():
            if type(m) is Conv and hasattr(m, 'norm'):
                m.conv = fuse_conv(m.conv, m.norm)
                m.forward = m.fuse_forward
                delattr(m, 'norm')

        return self
