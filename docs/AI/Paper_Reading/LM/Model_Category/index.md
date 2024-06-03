#### AE

autoencoder是一种 ^^对未标注数据进行无监督学习^^ 得到相应编码表示的神经网络。包含两个部分：

1. **encoder**：对输入数据进行加密转换（高纬度 → 低纬度的bottleneck）
2. **decoder**：对加密转换的数据进行解码还原（低纬度的bottleneck → 高纬度）
<div class="one-image-container">
    <img src="\AI\Paper_Reading\LM\Model_Category\images\autoencoder.png" style="width: 50%;">
    <p style="text-align: center;">AE网络示意</p>
</div>

#### DAE

1. Easy Data Augmentation的替换思想

    - **MASK**：token sequence 或 char image pixel
    - **ELECTRA模式**
    <div class="one-image-container">
        <img src="\AI\Paper_Reading\LM\Model_Category\images\ELECTRA.jpg" style="width: 80%;">
        <p style="text-align: center;">ELECTRA网络示意</p>
    </div>
    - **token sentnece repentation + white noise**：将离散的token增加连续的扰动
    - **char image + perturbations**：e.g., 白噪声, 几何变换 (旋转, 缩放, 平移，畸变，等)，其中常用的畸变操作有①同一字符的不同字体表述；②相似字符，如E和ε；③使用模型直接对字符进行畸变

2. Dropout
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


#### AR


#### VAE


#### VQVAE


#### CVAE


#### Difussion