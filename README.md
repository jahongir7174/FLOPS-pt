Counting FLOPS of PyTorch module

#### Usage

* MobileNetV2 is a baseline model implemented in `nn.py`
* `python main.py`

#### Optimized(Fused-BatchNorm)
```Number of parameters: 3487816```
```
Time per operator type:
        35.5653 ms.    91.6057%. Conv
        2.76578 ms.    7.12385%. Clip
       0.329736 ms.   0.849303%. FC
       0.140169 ms.   0.361034%. Add
       0.023362 ms.  0.0601736%. ReduceMean
        38.8243 ms in Total
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