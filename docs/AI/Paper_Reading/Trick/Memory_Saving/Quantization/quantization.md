量化是将包含更多信息的表示离散化为包含较少信息表示的过程，通过减少模型参数和激活值的表示精度以达到降低模型存储空间和计算量的目的。
### 量化流程
#### 数据格式

- S(ign): 符号位部分; 
- E(xponent): 指数位部分，指数位越多数值范围越大
- M(antissa): 也用Fraction表示，尾数位部分。尾数位越多精度越高

| 数据类型      | 符号位                          | 指数位 | 小数位
| ----------- | ------------------------------------ | --- | ---|
| FP64 | 1 | 11 | 52 |
| FP32       | 1  | 8 | 23 |
| TF32 | 1 | 8 | 10 |
| BF16    | 1 | 8 | 7 |
| FP16       | 1 | 5 | 10 |
| FP8 E4M3 | 1 | 4 | 3 |
| FP8 E5M2 | 1 | 5 | 2 |
| FP4 | 1 | 2 | 1 | 
| NF4 | 

> 浮点数转二进制格式可视化工具 [IEEE 754 Conventer](https://www.h-schmidt.net/FloatConverter/IEEE754.html)
#### Quantize
量化，将高精度数值类型转化为低精度数值类型，如`FP32 → FP16`




1. Block-wise k-bit quantization
    
    $$
    \begin{aligned}
    \text{quant} =& \text{round}(c^{\text{FP32}}\cdot X^{\text{FP32}}) \\
    \text{dequant} =& \frac{X^{\text{FP32}}}{c^{\text{FP32}}}
    \end{aligned}
    $$

> block: 为防止一次性量化过多元素，可以将$X\in\mathbb{R}^{b*h}$，每$n=(b*h)/B$ 个元素作为单个block统一进行量化，减缓由部分极值损害整体量化效果的现象    

$X^{Int8}=round(\frac{127}{absmax(X^{FP32})}X^{FP32})=round(c^{FP32}*X^{FP32})$
        - 为保留0值和左右边界-1和1，且最终含16个映射值  
        - 对称：0占两个index，正负数分别占7个index  
        - 非堆成：0占一个index，正数占8个index，负数占7个index  

#### Dequantize
解量化，将量化后的低精度数值类型转化为高精度数值类型，如`FP16 → FP32`

$X^{FP32}=dequant(c^{FP32}, X^{Int8})=\frac{X^{Int8}}{c^{FP32}}$
3. block-wise quantization：对于$X\in\mathbb{R}^{b*h}$，可以每$n=(b*h)/B$ 个元素统一进行量化，减少单次量化元素，减少部分量化值极少出现又被占的现象

### 量化方案
#### PTQ

#### QAT

1. data free，不适用校准集，直接量化，使用简单，但一般会带来较大精度损失
2. calibration，基于校准集，通过输入少量的真实数据进行统计分析
3. finetune，基于训练微调的方案，将量化胡茬在训练时自动调整权重，可带来更好的精度提升，但需要额外修改模型训练代码，开发周期较长

量化分类：

1. 二值化
2. 线性量化
3. 对数量化
- SmoothQuant: Accurate and Efficient Post-Training Quantization for Large Language Models


- [NF4](../../LLM_Extend/LLM_SFT/qlora.md): 本质上为4-bit字节码，分别表示下标0-15对应的量化值   










- $x=(-1)^S*2^{E-2^{\#E-1}+1}*1.M$
    - #E表示指数位的位数，指数位E的取值范围为[0, 2^#E]，最终取值为[-2^{\#E-1}+1, -2^{\#E-1}]
    - M计算方式：1） 去除后置0得到$M^{'}$；2）$M=\frac{M^{'}}{2^{\#M^{'}}}$
    - https://www.cnblogs.com/lemonzhang/p/17843336.html
    - https://blog.csdn.net/baoyan2015/article/details/136526423
    - https://zhuanlan.zhihu.com/p/676509123
    - https://mp.weixin.qq.com/s?__biz=MzI1MjQ2OTQ3Ng==&mid=2247618327&idx=2&sn=038c155d6082feab35789005c7cfc46e&chksm=e9e0069cde978f8a2251be0881acd894aeb6cf497e448cface31f052679edba0ed13d703ad15&scene=27
    - https://km.netease.com/v4/detail/blog/223053
    - https://zhuanlan.zhihu.com/p/666234324
    - https://readpaper.feishu.cn/docx/CrMGdSVPKow5d1x1XQMcJioRnQe

```python
torch.set_printoptions(precision=60)
a = torch.tensor(10**6, dtype=torch.float32)
> tensor(1000000.)
b = torch.tensor(10**6, dtype=torch.float16)
> tensor(inf, dtype=torch.float16)
```

- TF32: tensor float 32，为对齐FP32和FP16，实际只有19位
- BF16: brain float, 由google brain提出
- normalfloat 4-bit: NF4由华盛顿大学在QLoRA论文中提出
    - offset = (1 - 1/(2*15) + 1 - 1/(2*16))/2
    - 累计密度分位划分，分位点除以边界值归一化为[-1, 1]
    - https://onlinestatbook.com/2/calculators/normal_dist.html

