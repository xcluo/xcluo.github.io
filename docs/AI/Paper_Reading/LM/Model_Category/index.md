#### AE

Auto-Encoder是一种 ^^对未标注数据进行无监督学习^^ 得到相应编码表示的神经网络。整体是一个加密再重构的过程，即$p,q=\arg \min_{p,q} \Vert X - p[q(X)]\Vert$，包含两个部分：

1. **encoder**：对输入数据进行加密转换（高纬度 → 低纬度的bottleneck）
2. **decoder**：对加密转换的数据进行解码还原（低纬度的bottleneck → 高纬度）
<div class="one-image-container">
    <img src="\AI\Paper_Reading\LM\Model_Category\images\AE_schema.png" style="width: 50%;">
    <p style="text-align: center;">AE网络示意</p>
</div>

#### DAE
Denoising Auto-Encoder 在AE模型基础上加入噪声进行还原，负重训练提升模型鲁棒性

1. replace：对输入进行加噪
    - **MASK**：token sequence 或 char image pixel
    - **ELECTRA模式**
    <div class="one-image-container">
        <img src="\AI\Paper_Reading\LM\Model_Category\images\ELECTRA.jpg" style="width: 80%;">
        <p style="text-align: center;">ELECTRA网络示意</p>
    </div>
    - **token sentnece repentation + white noise**：将离散的token增加连续的扰动
    - **char image + perturbations**：e.g., 白噪声, 几何变换 (旋转, 缩放, 平移，畸变，等)，其中常用的畸变操作有①同一字符的不同字体表述；②相似字符，如E和ε；③使用模型直接对字符进行畸变

2. insert：对输入进行加噪
    - token sequence中插入标点、字符、空格、表情等 ^^不影响语义^^ 的特殊字符

3. Dropout：对模型中间的潜在表示进行加噪
    - **standard dropout**：[seq_len, dim] 所有值独立dropout
    - **spatial dropout**：[seq_len, dim] 以channel维度的值为单位进行dropout
<div class="row-image-container">
    <div>
        <img src="\AI\Paper_Reading\LM\Model_Category\images\standard_dropout.png">
        <p>standard dropout</p>
    </div>
    <div>
        <img src="\AI\Paper_Reading\LM\Model_Category\images\spatial_dropout.png">
        <p>spatial dropout</p>
    </div>
</div>

#### VAE
AE模型虽然能够学习到输入内容的representation，但是其各自的编码空间较为离散（如文本“中国”和“中，，，，国”的向量空间距离可能较大，又或【半月】和【满月】的图片向量空间距离也可能较大）。

而变分自编码器Variational Auto-Encoder通过encoder学习输入的中间表示分布 $(\mu, \sigma)$，随后基于分布进行采样$\mu + \sigma*e$再输入到decoder进行解码，最终学习到更加复杂且丰富的representation

<div class="one-image-container">
    <img src="\AI\Paper_Reading\LM\Model_Category\images\VAE_schema.png" style="width: 80%;">
    <p style="text-align: center;">VAE网络示意</p>
</div>
!!! info ""
    - $m$：预测的中间向量表示的均值$\mu$
    - $\sigma$：预测的中间向量表示的标准差取对数$\log\sigma$
    - $e$：从标准正态分布中采样的随机数
    - $L_1$：$D_{KL}( q(z|x)\Vert \mathcal{N})$，用于规范预测的噪声应趋近于加入的噪声分布（白噪声）
    - $L_2$：重构损失，如Cross-Entropy
    - 本质上也是对模型中间的潜在表示进行加噪

> https://arxiv.org/pdf/1312.6114v10

- [VAE in nlp](VAE/VAE_in_nlp.md)

#### CVAE
VAE存在一个问题，即无法指定**生成**指定的目标，因此就产生了条件变分自编码器Conditional Variational Auto-Encoder，其encoder和decoder有额外的输入——标签 $one\_hot\_y$


<div class="one-image-container">
    <img src="\AI\Paper_Reading\LM\Model_Category\images\CVAE_schema.jpg" style="width: 100%;">
    <p style="text-align: center;">CVAE网络示意</p>
</div>

> infer时 encoder输入为[randn(dim, ), label]

#### VQVAE
通常在AE $p[q(x)]=p(z)$ 中，我们只能看到原始数据 $x$，无法知晓 $z$，这也是其为什么称为数据的潜在表示。然而我们希望得到 $z$，因为它是数据更基础且压缩的表示。另外，潜在表示 $z$ 是很多算法很有用的输入。

理想情况下，我们希望隐空间中语义相似的数据点彼此相邻，而语义不同的点彼此远离。最好的情况是，大部分数据分布在隐空间中构成紧凑的空间，而不是无限大。变分量化自编码器Vector Quantized Variational Auto-Encoder 通过向网络添加离散的 codebook 组件来扩展标准自编码器。

<div class="one-image-container">
    <img src="\AI\Paper_Reading\LM\Model_Category\images\VQ-VAE_schema.png" style="width: 100%;">
    <p style="text-align: center;">VQ-VAE网络示意</p>
</div>

通过将连续的中间潜在表示离散化

   1. encoder输出$z_{e}(x) \in (H, W, dim)$
   2. $z_{e}(x)$ 与$codebook \in (codebook\_size, dim)$ 进行相似度计算，并 ^^选取后者相似度最高的特征进行替^^ 换得到 $z_q(x)\in (H, W, dim)$
   3. 输入decoder中进行重构

损失函数 $\mathcal{L}=\log{(p(x|q(x)))} + \Vert \text{sg}[z_e(x)]-e\Vert^2_2 + \beta\Vert z_e(x)-\text{sg}[e] \Vert^2_2$

   - $\text{sg}$ 表示stop gradient，即反向梯度计算到此为止
   - 第一项为重构损失
   - 第二项为codebook的对齐损失，只更新codebook
   - 第三项为encoder向codebook靠近的损失，只更新encoder

> https://arxiv.org/pdf/1711.00937'


#### AR
Auto-Regressive模型的输出结果依赖于其自生的先前值（如NLP中$X_{1,...,t-1}$）和一些其它环境因素确定的，整体是一个迭代输出的过程。
> 狭义上的AR模型输出的是线性结果，所以LLM的AR模型是一个广义的称呼概念
<div class="one-image-container">
    <img src="\AI\Paper_Reading\LM\Model_Category\images\AR_schema.jpg" style="width: 40%;">
    <p style="text-align: center;">AR网络示意</p>
</div>



#### Difussion