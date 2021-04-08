import os

import onnx
import torch
from caffe2.proto import caffe2_pb2
from caffe2.python import core, workspace, model_helper
from caffe2.python.onnx.backend import Caffe2Backend
from onnx import optimizer

from nets import nn


def torch2onnx():
    print("==> Creating PyTorch model")
    model = nn.MobileNetV2().fuse().eval()

    inputs = torch.randn((1, 3, 224, 224), requires_grad=True)
    model(inputs)

    print("==> Exporting model to ONNX format")
    dynamic_axes = {'input0': {0: 'batch'}, 'output0': {0: 'batch'}}

    _ = torch.onnx.export(model, inputs, 'weights/model.onnx',
                          export_params=True,
                          verbose=False,
                          input_names=["input0"],
                          output_names=["output0"],
                          keep_initializers_as_inputs=True,
                          dynamic_axes=dynamic_axes,
                          opset_version=10)

    print("==> Loading and checking exported model from ")
    onnx_model = onnx.load('weights/model.onnx')
    onnx.checker.check_model(onnx_model)  # assuming throw on error
    print("==> Done")


def traverse_graph(graph, prefix=''):
    content = []
    indent = prefix + '  '
    graphs = []
    num_nodes = 0
    for node in graph.node:
        pn, gs = onnx.helper.printable_node(node, indent, subgraphs=True)
        assert isinstance(gs, list)
        content.append(pn)
        graphs.extend(gs)
        num_nodes += 1
    for g in graphs:
        g_count, g_str = traverse_graph(g)
        content.append('\n' + g_str)
        num_nodes += g_count
    return num_nodes, '\n'.join(content)


def optimize_onnx():
    onnx_model = onnx.load('weights/model.onnx')
    passes = ['eliminate_identity',
              'eliminate_nop_dropout',
              'eliminate_nop_pad',
              'eliminate_nop_transpose',
              'eliminate_unused_initializer',
              'extract_constant_to_initializer',
              'fuse_add_bias_into_conv',
              'fuse_bn_into_conv',
              'fuse_consecutive_concats',
              'fuse_consecutive_reduce_unsqueeze',
              'fuse_consecutive_squeezes',
              'fuse_consecutive_transposes',
              'fuse_pad_into_conv', ]
    optimized_model = optimizer.optimize(onnx_model, passes)

    onnx.save(optimized_model, 'weights/model.onnx')


def onnx2caffe():
    print("==> Exporting ONNX to Caffe2 format")
    onnx_model = onnx.load('weights/model.onnx')
    caffe2_init, caffe2_predict = Caffe2Backend.onnx_graph_to_caffe2_net(onnx_model)
    caffe2_init_str = caffe2_init.SerializeToString()
    with open('weights/model.init.pb', "wb") as f:
        f.write(caffe2_init_str)
    caffe2_predict_str = caffe2_predict.SerializeToString()
    with open('weights/model.predict.pb', "wb") as f:
        f.write(caffe2_predict_str)
    print("==> Done")


def print_flops():
    print("==> Counting FLOPS")
    model = model_helper.ModelHelper(name="model", init_params=False)

    init_net_proto = caffe2_pb2.NetDef()
    with open('weights/model.init.pb', "rb") as f:
        init_net_proto.ParseFromString(f.read())
    model.param_init_net = core.Net(init_net_proto)

    predict_net_proto = caffe2_pb2.NetDef()
    with open('weights/model.predict.pb', "rb") as f:
        predict_net_proto.ParseFromString(f.read())
    model.net = core.Net(predict_net_proto)

    model.param_init_net.GaussianFill([],
                                      model.net.external_inputs[0].GetUnscopedName(),
                                      shape=(1, 3, 224, 224),
                                      mean=0.0,
                                      std=1.0)
    workspace.RunNetOnce(model.param_init_net)
    workspace.CreateNet(model.net)
    workspace.BenchmarkNet(model.net.Proto().name, 5, 100, True)
    print("==> Done")


def print_parameters():
    model = nn.MobileNetV2().fuse().eval()
    _ = model(torch.zeros(1, 3, 224, 224))
    params = sum(p.numel() for p in model.parameters())
    print('{:<20} {:<8}'.format('Number of parameters:', int(params)))


if __name__ == '__main__':
    if not os.path.exists('weights'):
        os.makedirs('weights')
    print_parameters()
    torch2onnx()
    optimize_onnx()
    onnx2caffe()
    print_flops()
