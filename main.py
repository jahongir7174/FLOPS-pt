import os
import warnings

warnings.filterwarnings("ignore")


def benchmark(model, shape):
    import os
    import onnx
    import torch
    from caffe2.proto import caffe2_pb2
    from caffe2.python.onnx.backend import Caffe2Backend
    from caffe2.python import core, model_helper, workspace

    inputs = torch.randn(shape, requires_grad=True)
    model.eval()
    model(inputs)

    # export torch to onnx
    _ = torch.onnx.export(model, inputs, './weights/model.onnx', True, False,
                          input_names=["inputs"],
                          output_names=["outputs"],
                          keep_initializers_as_inputs=True,
                          operator_export_type=torch.onnx.OperatorExportTypes.ONNX_ATEN_FALLBACK,
                          opset_version=10)

    onnx.checker.check_model(onnx.load('./weights/model.onnx'))

    # export onnx to caffe2
    onnx_model = onnx.load('./weights/model.onnx')

    caffe2_init, caffe2_predict = Caffe2Backend.onnx_graph_to_caffe2_net(onnx_model)

    # print benchmark
    model = model_helper.ModelHelper(name="model", init_params=False)

    init_net_proto = caffe2_pb2.NetDef()
    init_net_proto.ParseFromString(caffe2_init.SerializeToString())
    model.param_init_net = core.Net(init_net_proto)

    predict_net_proto = caffe2_pb2.NetDef()
    predict_net_proto.ParseFromString(caffe2_predict.SerializeToString())
    model.net = core.Net(predict_net_proto)

    model.param_init_net.GaussianFill([],
                                      model.net.external_inputs[0].GetUnscopedName(),
                                      shape=shape, mean=0.0, std=1.0)
    workspace.RunNetOnce(model.param_init_net)
    workspace.CreateNet(model.net)
    workspace.BenchmarkNet(model.net.Proto().name, 5, 100, True)
    # remove onnx model
    os.remove('./weights/model.onnx')


def main():
    if not os.path.exists('weights'):
        os.makedirs('weights')

    from nets.nn import MobileNetV2
    model = MobileNetV2().fuse()
    shape = (1, 3, 224, 224)

    benchmark(model, shape)


if __name__ == '__main__':
    main()
