量化是将包含更多信息的表示离散化为包含较少信息表示的过程，通过减少模型参数与激活值的表示精度以达到降低模型存储空间和计算量的目的。

1. 模型存储空间会减少，算力要求降低，，但推理速度不一定更快；
2. 模型效果效果会略微下降；

### 量化概念
#### 浮点数格式

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
    
#### 浮点数分类

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

!!! info ""
    - Nvidia专为Ampere架构设计的数据格式TF32(TensorFloat 32)，实际上只使用了19位
    - Google Brain提出BF16(BrainFloat 16)
    - 华盛顿大学在QLoRA中提出的[NF4](../../LLM_Extend/LLM_SFT/qlora.md)(NormalFloat 4)，本质上为4-bit字节码，对应0-15下标的固定浮点数

### Uniform Quantization
可通过缓存历史activation（如KV cache）和权重参数优化模型在LLM推理性能，即将  $Y = XW$ 通过量化数据 $\tilde{Y} = \tilde{X}\tilde{W} = \hat{X} \cdot s_X\hat{W} \cdot s_W $ 近似

1. **确定量化数值范围Range (Clipping)**，包括范围确定和范围裁剪，离群值可不进行量化处理也可clamp为边界值
    - ^^分位数统计法^^，$\text{quantile}\left(\vert x \vert, \text{cdf_percent}\right)$
    - ^^动量累计法^^，基于动量方法(running mean)计算activation range，如

        $$
        \begin{aligned}
            x_\text{max} =& \alpha x_\text{max} + (1-\alpha) \max\left(x_\text{current-iteration} \right) \\
            x_\text{min} =& \alpha x_\text{min} + (1-\alpha) \min\left(x_\text{current-iteration} \right) \\
        \end{aligned}
        $$

2. **量化Quantize**，将高精度数值表示转化为低精度数值表示，如`FP32 → FP16`；
3. **反量化Dequantize**，将量化后的低精度数值表示转化为高精度数值表示，如`FP16 → FP32`

$$
\begin{aligned}
s  =& \frac{\max \left(\vert x \vert \right)}{ 2^{b-1} - 1} \\
\hat{x}  =& \text{round}\left(\text{clamp}\left(\frac{x} {s}, -2^{b-1}, 2^{b-1}-1 \right)\right) \\
\tilde{x} =& \hat{x}\cdot s
\end{aligned}
$$

> WxAy 表示使用INT-x量化权重参数weight，INT-y量化激活值activation



### Non-Uniform Quantization
#### 二值化/三值化
元素只包含+1, -1（和0），虽然量化后权重值变得极端，但权重符号 Sign 往往比更精确的幅值 Magnitude 承载更多的信息，且神经网络在一定程度上对权重噪声具有鲁棒性，二值化/三值化可以视作一种极端的噪声。

$$
\hat{W}_{i, j} = \begin{cases}
 1 & \text{if }W_{i, j} \gt 0 \\
 0 & \text{if }W_{i, j}  = 0 \\
 -1  & \text{if }W_{i, j}  \lt 0
\end{cases}
$$

#### Logarithmic Quantization
当底数为2时，量化后的数值都是2的幂次。这使得复杂的乘法运算可以被简化为简单的位移（Bit-shift）操作。即传统乘法$y = wx$ 需要乘法器电路，功耗高，面积大；而对于对数量化后“乘法”，如果 $w = 2^n$，则 `y = x << n` 只需要移位器电路，极其高效。常见的量化方案：

1. 均匀对数量化（Uniform Logarithmic Quantization），直接将数值域映射到对数值域上
    - 确定量化范围 $[a, b]$ 和量化级别数 $K=2^b$
    - 在对数空间上均匀地创建 $K$ 个量化点
    - 将实数映射到最近的量化点上

    $$
    \begin{aligned}
        \hat{x} =& \text{sign}(x)\cdot 2^{L(x)} \\
        L(x) =& \text{round}\left(\log_2 \vert x \vert\right)
    \end{aligned}
    $$

    !!! info ""
        - 只保留指数位E和符号位S，该方法也称为Power-of-Two Quantization
        - $2^n$ 永远无法为0，因此需单独分配一个特定的码字codeword表示零，加上 $\text{sign}(x)$ ，共有$2^{b+1} - 1$ 种取值

2. 基于段的对数量化（Segment-Based Logarithmic Quantization），也被称为指数表示法量化，对浮点数指数位E和尾数位M两部分分别进行量化
    
    $$
    \begin{aligned}
        \hat{x} = \text{sign}(x)\cdot \left(1 + \frac{m}{2^M}\right) \cdot 2^e
    \end{aligned}
    $$

    > 该方法为 `float2float` 类型
    
#### K-means Quantization
K-means 量化将权重值看作数据点，使用 K-means 聚类算法找到最能代表这些数据分布的 $K$ 个中心点（聚类中心），然后将每个权重值量化为离它最近的那个中心点的值（值域为 $K$ 个点）。量化步骤：

- **应用K-means聚类**
    1. ^^初始化^^：选取 $K$ 个质心，如随机或基于直方图初始化
    2. ^^E step^^：计算离权值 $W_{i,j}$ 的最近质心
    3. ^^M step^^：基于聚类权值更新$K$ 个质心位置
    4. ^^重复步骤 b-c^^，直到收敛
- **构建量化映射表** 基于$K$个质心构建索引表 $\{idx: val\}$
- **执行量化** 实际上需只要保存量化值（所属质心）对应的索引，所需索引位数为 $\lceil \log_2 K \rceil$

> 常用于静态特性的Weight Quantization权重量化
#### Quantile Quantization
神经网络的权重和激活值通常不服从均匀分布，而是呈现出类似高斯分布或重尾分布的特点。线性量化（均匀量化）的主要缺点是它对所有区间的处理是平等的，忽略了数据的实际概率密度。分位数量化的核心思想是让每个量化区间包含大致相同数量的数据点，它通过确保每个量化箱（bin）的概率质量相等来实现这一目标。量化步骤

1. **构建经验累计分布函数ECDF**
2. **确定量化边界** 基于ECDF以及确定的$K$个量化区间数，确定量化边界
3. **分配量化值** 为每个bin分配代表值，根据数值关系确定所属bin，并分配相应的量化值
4. **执行量化** 找到量化值所属区间并存储相应的索引值，所需索引位数为 $\lceil \log_2 K \rceil$

典型工作  

- [NF4 quantization](../../LLM_Extend/PEFT/qlora.md) 浮点型分位量化

    $$
    \begin{aligned}
        s =& \max(\vert x \vert) \\
        \hat{x} =& \text{find_nearest_bin_value}\left(\frac{x}{c}\right) \\
        \tilde{x} = & \hat{x} \cdot s
    \end{aligned}
    $$


#### Sparse Non-uniform Quantization
稀疏非均匀量化结合了两种不同的模型压缩思想，稀疏化（Sparsity） 和非均匀量化（Non-uniform Quantization），核心思想是不平等地对待权重（或激活值）。



### 量化种类
#### AMP
Automatic Mixed Precision自动混合精度，对于模型总参数量$\Psi$，以Adam优化器下FP32/FP16混合精度训练为例,总空间消耗量为 $2\Psi + 2\Psi + 3*4\Psi$

1. FP16 Weight, FP16 Activation and FP16 Gradient
2. cast FP16 Gradient to FP32
3. 计算FP32 Momentum 和 FP32 Variance
4. 计算FP32 Gradient Update $\eta \frac{\hat{m}}{\sqrt{\hat{v}} + \epsilon}$  
    
    > 基于精度考量使用FP32进行计算，防止分子过小/分母过小导致梯度消失/梯度爆炸

5. cast FP32 Gradient Update to FP16
6. FP16精度下权重更新 `Weight = Weight - Gradient Update`


#### MPQ
Mixed-Precision Quantization混合精度量化

#### PTQ
Post Training Quantization，对训练后的模型进行量化处理，当量化为FP16时，无需校准；当量化为INT8时，一般需要使用少量代表性校准数据集，==不重新训练而是通过统计分析确定最优量化参数==，

- 权重参数直接量化，无需校准集，通常对称量化，也可非对称量化  
- 激活值需要通过校准集进行动态调整校正，可对称也可非对称  
- 混合精度量化，某些层高精度，某些层低精度

一般校准操作默认在层融合之前，用于校正激活值，校准步骤如下：

1. 准备校准数据集，数据规模一般为500~1000  
2. P模型推理，记录前向传播各层激活值activations  
3. 构建直方图**分别统计各层**激活值在不同数值区间的出现频率
    - 统计最小值和最大值，确定数值范围  
    - 将范围划分为若干个桶bin（如2048），统计各区间内激活值的出现频率  
4. 使用校准算法分析直方图  
    - 通过KL散度比较原始分布和量化后分布，寻找最小信息损失的截断阈值 (保留的最大FP32值，即`x=min(x, threshold), fp32_dist = hist[:threshold] / np.sum(hist[:threshold])`) 
    - 根据阈值计算每个量化后bin的宽度 `scale = threshold / INT8_max`，即`quant_bins = np.liespace(0, threshold, INT8_max)`  
    - 统计FP32激活值在上述量化桶的出现频数，并归一化频率`quant_dist /= np.sum(quant_dist)`  
    - 为避免log 0，`fp32_dist = np.clip(fp32_dist, 1e-10, 1), quant_dist=np.clip(quant_dist, 1e-10, 1)`
    - 计算KL distance
    - 选定range_width（对左右边界进行截断选择），然后进行 `bin_width = range_width / INT8_max` 分桶  
    - 计算归一化范围进行分桶 `(x - min(x)) / bin_width`
    - KL矫正算法：计算并选择最小量化前后KL散度对应的截断范围，适用于长尾分布、双峰分布等  
    - MinMax：直接取min/max值以**全量保存激活值动态范围**，因此对离群机值点十分敏感，适用于均匀分布情况  
    - MinMax改进，使用分位数截断缓解离群点影响，如$[P_{0.1\%}, P_{99.9\%}]$
    - 范围无需对称，因为可通过减去 `min_value` 进行平移处理

```python
import numpy as np
from scipy.stats import entropy

def compute_scale(activations, num_bins=2048):
    hist, bin_edges = np.histogram(activations, bins=num_bins)
    bin_width = bin_edges[1] - bin_edges[0]
    
    # 使用KL散度选择最优截断阈值
    def kl_divergence(threshold):
        # 将FP32分布截断到[0, threshold]并归一化，因为此处时ReLU激活函数示例
        fp32_dist = hist[:threshold] / np.sum(hist[:threshold])
        
        # 生成量化后的分布（INT8模拟）
        quant_bins = np.linspace(0, threshold, 256)
        quant_dist = np.zeros(256)
        for i in range(256):
            start = quant_bins[i]
            end = quant_bins[i+1] if i < 255 else threshold
            quant_dist[i] = np.sum(hist[(bin_edges >= start) & (bin_edges < end)])
        quant_dist /= np.sum(quant_dist)
        
        # 避免log(0)
        fp32_dist = np.clip(fp32_dist, 1e-10, 1)
        quant_dist = np.clip(quant_dist, 1e-10, 1)
        
        return entropy(fp32_dist, quant_dist)
    
    # 搜索最佳阈值
    best_threshold = np.argmin([kl_divergence(t) for t in range(100, num_bins)])
    scale = bin_edges[best_threshold] / 127.0  # INT8对称量化
    
    return scale
```

不同batch顺序会产生不同的校准尺度，因此建议使用large batch
    - https://docs.nvidia.com/deeplearning/tensorrt/archives/tensorrt-843/pdf/TensorRT-Developer-Guide.pdf
    - 常见校准算法如下


1. **Entropy Calibration**，选择对应量化前后分布最小KL散度的阈值来确定最优的量化参数

    $$
    D_{KL}(P\Vert Q) = \sum_{i} P(i)\log \frac{P(i)}{Q(i)}
    $$

    > $P$ 为原始模型，$Q$ 为量化后模型

2. **MinMax Calibration**，更适合NLP任务
3. **Legacy Calibration**，最小化量化前后的均方差


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

#### QKD
Quantization-aware Knowledge Distillation，将原始模型作为teacher model、量化后的模型作为student model进行蒸馏

### 量化粒度
#### Layer-wise Quantization
Layer-wise Quantization 是一种基础的模型量化方法，它将神经网络中每一层的权重或激活值作为一个整体进行量化（使用统一的量化参数缩放因子和零点）

- 粒度较为粗糙，精度损失较大

#### Block-wise Quantization
Block-wise k-bit Quantization将数据划分成 blocks（如$8\times 8$），对于每个block独立应用INT-k量化方案

- [x] 分块独立处理能有效减缓由极端极值 $\max(\vert x \vert)$ 导致量化缩放因子 $s_{x, k}$ 过大，表现量化结果过于集中模糊，损害整体量化效果
- 转为Tensor Core设计


#### Group-wise Quantization
Group-wise Quantization将数据按通道方向划分成 groups（$W\in \mathbb{R}^{d_{in}\times d_{out}}$ 按行维度或列维度划分，一般基于GEMM $Y=XW$特性采用Column-wise拆分），对于每个group独立应用INT-k量化方案

- 更适用于Attention中的注意力权重计算，与GEMM兼容
- 通用于CPU/GPU硬件架构

#### Token-wise Quantization
Token-wise Quantization 是一种面向动态序列数据（如Transformer的输入和激活值）的细粒度量化方法，其核心思想是为输入序列中的每个 Token 独立计算量化参数，从而显著提升低比特量化下的模型精度。

- 在推理阶段会因为实时计算 $s_t$ 而引入额外开销，但一般＜5%



