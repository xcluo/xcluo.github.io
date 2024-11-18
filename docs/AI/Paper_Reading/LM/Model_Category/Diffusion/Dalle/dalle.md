DALLE，名称来源于著名画家达利（**Da**lí）和机器人总动员（Wa**ll-E**），是Open AI推出的一个可以根据书面文字生成图像的人工智能系统。

## DALL-E-1

## DALL-E-2
也称作unCLIP，即CLIP的逆过程：{>>通过给定imge_hidden_state生成图片<<}。层级式地生成64\*64 -> 256\*256 -> 1024\*1024分辨率的图片，逐渐高清
### Framework
<div class="one-image-container">
    <img src="\AI\Paper_Reading\LM\Model_Category\Diffusion\Dalle\image\dalle-2_framework.png" style="width: 100%;">
    <p style="text-align: center;">unCLIP概略图</p>
</div>

DALLE-2模型包含**prior**和**decoder**两个模块，前者基于输入的文本生成相应的图片特征，后者则根据图-文特征生成图片，模型包含的主要技术有：

1. 对比学习（CLIP训练时使用的方法）
2. 多模态（图-文模态信息信息交互、转换）
3. 扩散模型（prior以及decoder模块主要思路）

#### prior
输入描述文本，输出相应的clip imgae hidden state特征


- **train**
    1. 输入图文对 ($x, y$) 至 frozen CLIP 中获取对应的图片特征 $z_x$ 和 文本描述特征 $z_y$
    2. 以文本信息 $y$ 和 $z_y$ 作为输入，训练prior使其能够将文本信息转化为对应的图片特征 $z_x$ (作为训练的ground-truth)，prior有两种实现方案
        - **autoregressive prior**：$\text{gpt}_\theta(y, z_y)=\hat{z_x}$
        - **diffusion prior**：扩散模型去噪还原得到 $z_x$，$f_\theta(y, z_y, t, z_x^{(t)})=\hat{z_x}$ ，每个阶段不预测加入的噪声而是直接预测文本描述对应的图片特征
    3. $\mathcal{L}=\mathbb{E}_{t\sim[1,T], z_i^{(t)}\sim q(t)}\Big[\Vert f_\theta(z_i^{(t)}, t, y) - z_i\Vert^2\Big]$
    !!! info ""
        - 10%的训练时间对 $z_y$ 进行drop，50%的时间对文本描述进行mask
        - 不同于图片扩散，对于图片 ==特征的扩散还原选用了Transformer模型== 而不是UNet ，即输入[`encoding_y`, `clip_text_embedding_y`, `time_embedding_of_t`, `x_t`, `noised_image_embedding`, `[IMG]`]，模型最后层 `[IMG]` 的值即为预测的$\hat{z_x}$

- **infer**
    1. 输入文本 $y$ 至 frozen CLIP得到 $z_y$
    2. 初始化 $z_x^{(T)}\sim\mathcal{N}\text{(}0, \text{I}\text{)}$
    3. $f_\theta(y, z_y, t, z_x^{(t)})=\hat{z_x}$，通过 $\hat{z_x}$ 和 $z_x^{(t)}$ 得到 $t-1$ 时刻的图像分布，并采样得到 $z_x^{(t-1)}$
    4. 重复第3步直至获取 $z_x^{(0)}$
    !!! info ""
        autogressive prior 直接一步预测得到 $\hat{z_x}$

#### decoder
- **train & sample**
    1. 正常的图像diffusion过程，$f_\theta(y, z_y, \hat{z_x}, t, x_t)=\hat\epsilon$
    2. 模型为UNet
    !!! info ""
        - 训练时5%的训练时间对 $z_x$ 进行了drop

### Result
#### Importance of prior
<div class="one-image-container">
    <img src="\AI\Paper_Reading\LM\Model_Category\Diffusion\Dalle\image\dalle-2_prior_ablation.jpg" style="width: 90%;">
    <p style="text-align: center;">第一行：$decoder(y)$；第二行：$decode(y,z_y)$；第三行：$decoder(ar\_piror(y, z_y))$ </p>
</div>


<div class="one-image-container">
    <img src="\AI\Paper_Reading\LM\Model_Category\Diffusion\Dalle\image\dalle-2_prior_ablation_quantify.jpg" style="width: 90%;">
    <p style="text-align: center;">ar_prior 和 diffusion_prior 效果对比（人工评审结果）</p>
</div>

> 比起直接输入文本特征进行图像还原，<span style="color: green;">prior能够增强文本生成图像的效果，且 diffusion_prior 比 ar_prior 得到的图片特征在生成图片表现更加优越</span>

#### 效果对比
<div class="row-image-container">
    <div>
        <img src="\AI\Paper_Reading\LM\Model_Category\Diffusion\Dalle\image\dalle-2&glide_perference.jpg" style="width: 77%">
        <p>unCLIP & GLIDE不同尺度下生成图片受人喜爱比例</p>
    </div>

    <div>
        <img src="\AI\Paper_Reading\LM\Model_Category\Diffusion\Dalle\image\dalle-2&glide_fid.jpg">
        <p>unCLIP & GLIDE不同尺度下生成图片FID</p>
    </div>

</div>


<div class="one-image-container">
    <img src="\AI\Paper_Reading\LM\Model_Category\Diffusion\Dalle\image\dalle-2_zero-shot_effect.jpg" style="width: 90%;">
    <p style="text-align: center;">unCLIP与同类工作效果对比</p>
</div>

<div class="one-image-container">
    <img src="\AI\Paper_Reading\LM\Model_Category\Diffusion\Dalle\image\dalle-2_sample_on_MSCOCO.jpg" style="width: 100%;">
    <!-- <p style="text-align: center;">unCLIP与同类工作效果对比</p> -->
</div>

> <span style="color: green;">unCLIP 在效果和数值测试表现中均取得最好结果</span>

#### Application
- 文本生成图片
<div class="row-image-container">
    <div>
        <img src="\AI\Paper_Reading\LM\Model_Category\Diffusion\Dalle\image\dalle-2_shiba.jpg"  style="width: 90%;">
        <!-- <p>unCLIP & GLIDE不同尺度下生成图片受人喜爱比例</p> -->
    </div>

    <div>
        <img src="\AI\Paper_Reading\LM\Diffusion\Dalle\image\dalle-2_panda.jpg" style="width: 90%;" >
        <!-- <p>unCLIP & GLIDE不同尺度下生成图片受人喜爱比例</p> -->
    </div>

    <div>
        <img src="\AI\Paper_Reading\LM\Diffusion\Dalle\image\dalle-2_teddy.jpg" style="width: 90%;" >
        <!-- <p>unCLIP & GLIDE不同尺度下生成图片受人喜爱比例</p> -->
    </div>
</div>

- 制作动画、渐变
<div class="one-image-container">
    <img src="\AI\Paper_Reading\LM\Model_Category\Diffusion\Dalle\image\dalle-2_text_diffs.jpg" style="width: 80%;">
    <!-- <p style="text-align: center;">unCLIP与同类工作效果对比</p> -->
</div>

- 图片重构（将图片的clip image embedding作为decoder输入）
<div class="one-image-container">
    <img src="\AI\Paper_Reading\LM\Model_Category\Diffusion\Dalle\image\dalle-2_change_ui.jpg" style="width: 80%;">
    <!-- <p style="text-align: center;">unCLIP与同类工作效果对比</p> -->
</div>

- 图片风格融合（将两张图片的clip image embedding加权作为decoder输入）
<div class="one-image-container">
    <img src="\AI\Paper_Reading\LM\Model_Category\Diffusion\Dalle\image\dalle-2_interpolation.jpg" style="width: 80%;">
    <!-- <p style="text-align: center;">unCLIP与同类工作效果对比</p> -->
</div>


#### Limitation
- 局限性

<div class="one-image-container">
    <img src="\AI\Paper_Reading\LM\Model_Category\Diffusion\Dalle\image\dalle-2_cube_image.jpg" style="width: 90%;">
    <!-- <p style="text-align: center;">unCLIP与同类工作效果对比</p> -->
</div>


<div class="one-image-container">
    <img src="\AI\Paper_Reading\LM\Diffusion\Dalle\image\dalle-2_mix_up_objects&attributes.jpg" style="width: 90%;">
    <!-- <p style="text-align: center;">unCLIP与同类工作效果对比</p> -->
</div>

> <span style="color: red">由于以CLIP为学习目标，集成了其只关注物体是否存在而不关注物体属性（e.g., 位置，角度，大小，颜色等）的局限性</span>

<div class="one-image-container">
    <img src="\AI\Paper_Reading\LM\Model_Category\Diffusion\Dalle\image\dalle-2_not_precisely_encoder_spelling.jpg" style="width: 90%;">
    <!-- <p style="text-align: center;">unCLIP与同类工作效果对比</p> -->
</div>

> <span style="color: red">CLIP embedding无法准确地解析文本的拼写信息（大概率是因为BPE编码基于统计而不是基于语义实现的）</span>

### Discovering
#### Hidden Vocabulary



<div class="one-image-container">
    <img src="\AI\Paper_Reading\LM\Model_Category\Diffusion\Dalle\image\dalle-2_language_farmer&talk.png" style="width: 100%;">
    <!-- <p style="text-align: center;">unCLIP概略图</p> -->
</div>
- prompt：描述两个老农讨论在蔬菜并生成标题
- 生成图片：<span style="color: orange">标题为蔬菜</span>，<span style="color: red">谈话内容为鸟</span>


<div class="one-image-container">
    <img src="\AI\Paper_Reading\LM\Model_Category\Diffusion\Dalle\image\dalle-2_language_birds&bugs.png" style="width: 100%;">
    <!-- <p style="text-align: center;">unCLIP概略图</p> -->
</div>
- prompt**(一堆鸟语)**：“Apoploe vesrreaitais eating Contarra
ccetnxniams luryca tanniounons”
- 生成图片：鸟吃虫

<div class="one-image-container">
    <img src="\AI\Paper_Reading\LM\Model_Category\Diffusion\Dalle\image\dalle-2_language_whale&talk.png" style="width: 100%;">
    <!-- <p style="text-align: center;">unCLIP概略图</p> -->
</div>
- prompt**(一堆鸟语)**：“Wa ch zod ahaakes rea.”
- 生成图片：<span style="color: green">海鲜</span>

!!! danger ""
    由于自然语言细节的表达无法保证，DALLE语言存在被“破解”用作非法用途的风险




## DALL-E-3

1. predict_start_from_noise，求$x_0$，$x_t = ax_0 + b\epsilon$
3. predict_noise_from_start，求$\epsilon$，$x_t = ax_0 + b\epsilon$
4. q_sample，求$x_t$，$x_t = ax_0 + b\epsilon$
5. q_posterior，求$\mu_{t-1}$，$\sigma_{t-1}$
2. predict_start_from_v，
3. calculate_v