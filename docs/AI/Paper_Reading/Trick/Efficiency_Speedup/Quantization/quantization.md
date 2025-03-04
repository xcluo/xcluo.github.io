量化是将包含更多信息的表示离散化为包含较少信息表示的过程，通过减少模型参数和激活值的表示精度以达到降低模型存储空间和计算量的目的。

1. 模型存储空间会减少，算力要求降低，，但推理速度不一定更快；
2. 模型效果效果会略微下降；
### 量化流程
#### Float Point Format

<div class="one-image-container">
    <img src="image/fp32_float_point_format.jpg" style="width: 95%;">
    <!-- <p>LoRA在Attention各部分权重上的消融实验效果</p> -->
</div>

- S(ign): 符号位部分; 
- E(xponent): 指数位部分，位数越多数值范围越大
- M(antissa): 也用Fraction表示，小数/尾数位部分，位数越多精度越高
- 浮点数转二进制格式可视化工具 [IEEE 754 Conventer](https://www.h-schmidt.net/FloatConverter/IEEE754.html)

    $$
    fp=(-1)^s*2^{e-bias}*(1 + m)
    $$

    > $bias = 2^{E-1} - 1$，因此指数部分值域为$[-2^{E-1}+1, 2^{E-1}]$  
    > $m = \sum_{n=1}^{M1} \text{bit}_n*2^{-n}$
    
#### Float Type

| Type      | S                          | E | M/F
| ----------- | ------------------------------------ | --- | ---|
| FP64 | 1 | 11 | 52 |
| FP32 | 1  | 8 | 23 |
| TF32 | 1 | 8 | 10 |
| BF16 | 1 | 8 | 7 |
| FP16 | 1 | 5 | 10 |
| FP8 E4M3 | 1 | 4 | 3 |
| FP8 E5M2 | 1 | 5 | 2 |
| FP4 | 1 | 2 | 1 | 

> - Nvidia专为Ampere架构设计的数据格式TF32，即TensorFloat 32，实际上只使用了19位
> - Google Brain提出的BF16，即BrainFloat 16
> - 华盛顿大学在QLoRA中提出的[NF4](../../LLM_Extend/LLM_SFT/qlora.md)，即NormalFloat 4，本质上为4-bit字节码，用于下标0-15对应的固定浮点数



#### Quantize & Dequantize
量化，将高精度数值类型转化为低精度数值类型，如`FP32 → FP16`；解量化，将量化后的低精度数值类型转化为高精度数值类型，如`FP16 → FP32`

1. Float2Float quantization，转型量化
高精度转化为低精度时需要在指数位调整偏移量、尾数位低位截断；低精度转化为高精度时则需要在指数位调整偏移量、尾数位低位进行补0操作，常用 `cast` 方法实现转换

2. block-wise k-bit (symmetric) quantization，对称线性量化  
将数据划分为多个块block（元素个数为$B$），然后在每个块内应用k-bit量化方案

    $$
    \begin{aligned}
    c(x_{up}, k)  =& \frac{2^{k-1} - 1}{\text{absmax}(x_{up})} \\
    q(x_{up}, k, c) =& round(x_{up}*c) \\
    deq(x_{low}, k, c) =& \frac{x_{low}}{c}
    \end{aligned}
    $$

    > 分块原因：减缓由部分极值导致量化缩放因子 $c$ 过小，导致量化结果过于集中，粒度模糊，损害整体量化效果的现象

1. asymmetric quantization，非对称线性量化  

    $$

    $$

2. [NF4 quantization](../../LLM_Extend/LLM_SFT/qlora.md)，分位量化  
先验地认为模型数据符合正态分布，并基于正态分布累计分布函数CDF求得16个固定浮点数作为最终量化值；此外，还设计了多重量化方案，即进一步对多个量化缩放因子进行浮点数量化


    $$
    \begin{aligned}
        c(x_{up}) =& \frac{1}{\text{absmax}(x_{up})} \\
        q(x_{up}, c) =& \text{find_nearest}(x_{up}*c) \\
        deq(x_{low}, c) = & \frac{x_{low}}{c}
    \end{aligned}
    $$



### 量化方案
#### AMP
混合精度训练[Mixed Precision Training](https://arxiv.org/pdf/1710.03740)

#### PTQ
Post Training Quantization


converter.representative_dataset

```python
converter.optimizations = [tf.lite.Optimize.DEFAULT]
# 动态范围量化，缺省状态

# 半精度量化 Float16 Quantization: 仅量化权重
converter.target_spec.supported_types = [tf.float16]

# 全整型量化 Full Integer Quantization: 权重和激活值都进行量化
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
converter.inference_input_type = tf.uint8   # or tf.int8
converter.inference_output_type = tf.uint8  # or tf.int8

# int8权重float16激活
converter.target_spec.supported_ops = [tf.lite.OpsSet.EXPERIMENTAL_TFLITE_BUILTINS_ACTIVATIONS_INT16_WEIGHTS_INT8]
```
#### QAT
Quantization Aware Training

1. data free，不适用校准集，直接量化并输出，但一般会带来较大精度损失
2. calibration，基于校准集，通过输入少量的真实数据进行统计分析来矫正量化数据
3. finetune，基于训练微调的方案，将量化胡茬在训练时自动调整权重，可带来更好的精度提升，但需要额外修改模型训练代码，开发周期较长


- TensorFlow Lite Optimizing Converter(Toco) 转换器

量化分类：

1. 二值化
2. 线性量化
3. 对数量化
- SmoothQuant: Accurate and Efficient Post-Training Quantization for Large Language Models
