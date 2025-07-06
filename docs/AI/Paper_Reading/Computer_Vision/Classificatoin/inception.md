## Inception-1
> 论文：Going deeper with convolutions  
> Google & University of North Carolina & University of Michigan, 2014 Sep, CVPR 2015


### 主要内容
### Inception
![alt text](image/inception-1_module.png)
1. Naive Version：[1x1 conv, 3x3 conv, 5x5 conv]
2. Dimension Reduction Version：[1x1 conv, 1x1 conv + 3x3 conv, 1x1 conv + 5x5 conv, 3x3 max-pooling + 1x1 conv]
- based on the Hebbian principle and the intuition of multi-scale processing
- #3x3 reduce表示前置的1x1 conv filter数目
- pool proj表示3x3 max-pooling后的1x1 conv filter数目
- reduction/projection
    - input: `(N, H, W, C_in)`
    - 1x1 filter: `(1, 1, C_in, C_out)`
    - output: `(N, H, W, C_out)`
    - 当 `C_in > C_out`即reduction，反之为projection
- All these reduction/projection layers use rectified linear activation as well.
- filter sizes 1x1, 3x3 and 5x5, however this decision was based more on convenience rather than necessity.
- higher layers, their spatial concentration is expected to decrease suggesting that the ratio of 3x3 and 5x5 convolutions should increase as we move to higher layers.
- 少通道进行压缩：1x1 convolutions are used to compute reductions before the expensive 3x3 and 5x5 convolutions.
- occasional max-pooling layers with stride 2 to halve the resolution of the grid.
- start using Inception modules only at higher layers while keeping the lower layers in traditional convolutional fashion
- Patch Size=kernel size
- 深度拼接=channel dimension concatenation
#### GoogLeNet
- AveragePool 7x7+1(V) 和 Conv 1x1+1(V) 中的V表示valid填充，+\d后面的\d表示步长
- Conv 1x1+1(S) 中的S表示same填充
- 训练时三个分类器的损失加权求和（主分类器权重1，辅助分类器各0.3）,测试时只使用主分类器


## Inception-2
## Inception-3
## Inception-4


## Xception