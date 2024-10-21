- Quantize，量化，如将FP32转化为int8
- Dequantize，逆向量化，如将int8转化为FP32

量化是将输入从包含更多信息的表示离散化为包含较少信息的表示的过程，且为了充分使用更少位数据类型的整个取值范围，通常通过对输入数据进行归一化，以重新调整为目标数据类型范围

1. 量化
$X^{Int8}=round(\frac{127}{absmax(X^{FP32})}X^{FP32})=round(c^{FP32}*X^{FP32})$
2. 逆向量化
$X^{FP32}=dequant(c^{FP32}, X^{Int8})=\frac{X^{Int8}}{c^{FP32}}$
3. block-wise quantization：对于$X\in\mathbb{R}^{b*h}$，可以每$n=(b*h)/B$ 个元素统一进行量化，减少单次量化元素，减少部分量化值极少出现又被占的现象


量化方案：

1. data free，不适用校准集，直接量化，使用简单，但一般会带来较大精度损失
2. calibration，基于校准集，通过输入少量的真实数据进行统计分析
3. finetune，基于训练微调的方案，将量化胡茬在训练时自动调整权重，可带来更好的精度提升，但需要额外修改模型训练代码，开发周期较长

量化分类：

1. 二值化
2. 线性量化
3. 对数量化


- SmoothQuant: Accurate and Efficient Post-Training Quantization for Large Language Models

| 数据类型      | 符号位                          | 指数位 | 小数位
| ----------- | ------------------------------------ | --- | ---|
| FP32       | 1  | 8 | 23 |
| FP16       | 1 | 5 | 10 |
| BF16    | 1 | 8 | 7 |
| FP4 | 1 | 2 | 1 | 

BF16: brain float, 由google brain提出
4-bit normalfloat: 由华盛顿大学在QLoRA论文中提出