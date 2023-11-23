FLOPS counter using PyTorch

#### Installation

```
conda create -n FLOP python=3.8.10
conda activate FLOP
conda install pytorch torchvision torchaudio cudatoolkit=11.1 -c pytorch-lts -c nvidia
pip install onnx==1.8.1
pip instal future==0.18.3
pip install protobuf==3.20
pip install numpy==1.23.4
```

#### Usage

* MobileNetV2 is a baseline model implemented in `nn.py`
* `python main.py`

#### FLOPS and Parameters

```
Number of parameters: 3487816
Time per operator type:
        21.9012 ms.    89.5336%. Conv
        2.11011 ms.    8.62627%. Clip
       0.332954 ms.    1.36114%. FC
       0.101426 ms.   0.414637%. Add
      0.0136423 ms.  0.0557708%. AveragePool
     0.00209721 ms. 0.00857355%. Flatten
        24.4614 ms in Total
FLOP per operator type:
       0.598989 GFLOP.    99.5385%. Conv
       0.002561 GFLOP.   0.425581%. FC
    0.000216384 GFLOP.  0.0359582%. Add
       0.601766 GFLOP in Total
Feature Memory Read per operator type:
        35.8909 MB.    83.9532%. Conv
        5.12912 MB.    11.9976%. FC
        1.73107 MB.    4.04918%. Add
        42.7511 MB in Total
Feature Memory Written per operator type:
        26.7124 MB.    96.8475%. Conv
       0.865536 MB.    3.13805%. Add
          0.004 MB.  0.0145022%. FC
         27.582 MB in Total
Parameter Memory per operator type:
        8.75904 MB.    63.0917%. Conv
          5.124 MB.    36.9083%. FC
              0 MB.          0%. Add
         13.883 MB in Total
```