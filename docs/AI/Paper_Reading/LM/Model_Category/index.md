#### AE

Auto-Encoder是一种 ^^对未标注数据进行无监督学习^^ 得到相应编码表示的神经网络。整体是一个加密再还原的过程，包含两个部分：

1. **encoder**：对输入数据进行加密转换（高纬度 → 低纬度的bottleneck）
2. **decoder**：对加密转换的数据进行解码还原（低纬度的bottleneck → 高纬度）
<div class="one-image-container">
    <img src="\AI\Paper_Reading\LM\Model_Category\images\autoencoder.png" style="width: 50%;">
    <p style="text-align: center;">AE网络示意</p>
</div>

#### DAE
Denoising Auto-Encoder 在AE模型基础上加入噪声进行还原，负重训练提升模型鲁棒性

1. Easy Data Augmentation的replace

    - **MASK**：token sequence 或 char image pixel
    - **ELECTRA模式**
    <div class="one-image-container">
        <img src="\AI\Paper_Reading\LM\Model_Category\images\ELECTRA.jpg" style="width: 80%;">
        <p style="text-align: center;">ELECTRA网络示意</p>
    </div>
    - **token sentnece repentation + white noise**：将离散的token增加连续的扰动
    - **char image + perturbations**：e.g., 白噪声, 几何变换 (旋转, 缩放, 平移，畸变，等)，其中常用的畸变操作有①同一字符的不同字体表述；②相似字符，如E和ε；③使用模型直接对字符进行畸变

2. Easy Data Augmentation的insert
    - token sequence中插入标点、字符、空格、表情等 ^^不影响语义^^ 的特殊字符

3. Dropout
    - **standard dropout**：[seq_len, dim] 所有值独立dropout
    - **spatial dropout**：[seq_len, dim] 以dim维度的值为单位进行dropout
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
变分自编码器Variational Auto-Encoder通过encoder学习输入的中间表示 $(\mu, \sigma)$ 再进行采样输入到decoder

#### VQVAE
变分量化自编码器Variational Quantized Variational Auto-Encoder


#### CVAE
条件变分自编码器Conditional Variational Auto-Encoder


#### AR
Auto-Regressive模型的输出结果依赖于其自生的先前值（如NLP中$X_{1,...,t-1}$）和一些其它环境因素确定的，整体是一个迭代输出的过程。
> 狭义上的AR模型输出的是线性结果，所以LLM的AR模型是一个广义的称呼概念
<div class="one-image-container">
    <img src="\AI\Paper_Reading\LM\Model_Category\images\autoregressive.jpg" style="width: 40%;">
    <p style="text-align: center;">AR网络示意</p>
</div>




#### Difussion