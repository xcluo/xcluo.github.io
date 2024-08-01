### Vanilla VAE in nlp
> 论文：[Generating Sentences from a Continuous Space](pdf/vanilla_vae.pdf)  
> Google Brain, CoNLL, 2016

#### 模型架构
<div class="one-image-container">
    <img src="\AI\Paper_Reading\LM\Model_Category\VAE\image\vanilla_vae_framework.png" style="width: 80%;">
    <p style="text-align: center;">pipeline: encoder + VAE + decoder</p>
</div>

#### 工作要点
1. introduce `KL cost annealing` to deal with posterior collapse (also named KL-vanishing) 【$qp(z|x)\rightarrow p(z)$】</li>
    - 散度消失，此时从简单的先验p(z)中采样潜变量z,也能很好地重构出输入x，导致z携带的信息很少


<div class="one-image-container">
    <img src="\AI\Paper_Reading\LM\Model_Category\VAE\image\KL_divergence_vae_train.png" style="width: 80%;">
    <!-- <p style="text-align: center;">pipeline: encoder + VAE + decoder</p> -->
</div>


实验发现，VAE训练前期$\mathcal{L}_{KL}$激增，若无对应加权方案将使decoder拟合$p(z)$而不是$q(z|x)$，造成后验坍塌，可通过增加权重参数进行加权 $\beta*\mathcal{L}_{KL}$，例如：

- 0 -> linear/exp increase -> 1
- learnable $\beta$

</br>
2. latent space learned by VAE is more informative

<div class="one-image-container">
    <img src="\AI\Paper_Reading\LM\Model_Category\VAE\image\vanilla_vae_informative_latent_space.png" style="width: 80%;">
    <p style="text-align: left;">Penn Treebank language modeling results. NLL: negative log likelihood; Inputless: decoder($x_i|z,t_i$) without $x_{i-1}$</p>
</div>

</br>
3. latent space learned by VAE is much smoother

- enable homotopy (linear interpolation $z=z_1*(1-t)+z_2*t$)

<div class="row-image-container">
    <div>
        <img src="\AI\Paper_Reading\LM\Model_Category\VAE\image\ae_interpolation.png">
        <!-- <p></p> -->
    </div>
    <div>
        <img src="\AI\Paper_Reading\LM\Model_Category\VAE\image\vanilla_vae_interpolation.png" style="width: 80%;">
        <!-- <p>spatial dropout</p> -->
    </div>
</div>

由interpolation现象可知，传统AE模型生成句向量过于尖锐与离散，而VAE生成的句向量更加平滑且更保留更多的语法、主题以及句法特征信息。

### Optimus
> 论文：[OPTIMUS: Organizing Sentences via Pre-trained Modeling of a Latent Space](pdf/OPTIMUS.pdf)  
> **O**rganizing sentences via **P**re-**T**ra**i**ned **M**odeling of a **U**niversal **S**pace  
> MSR, EMNLP 2020
#### 模型架构

<div class="one-image-container">
    <img src="\AI\Paper_Reading\LM\Model_Category\VAE\image\OPTIMUS_framework.png" style="width: 80%;">

    <img src="\AI\Paper_Reading\LM\Model_Category\VAE\image\OPTIMUS_framework_decoder.png" style="width: 80%;">
</div>

1. $W_Mz=h_{memory}\in \mathbb{R}^{L\times H}$，类似于连续型prompt分别与transformer每层的hidden state进行concate
2. $W_Dz=h_{embedding}\in \mathbb{R}^H$，只在embedding layer生效，类似于$Emb_{token}, Emb_{pos}$
#### 工作要点
1. bridge the gap between pretrained encoder(e.g., BERT) and decoder(e.g., GPT-2)

<div class="one-image-container">
    <img src="\AI\Paper_Reading\LM\Model_Category\VAE\image\OPTIMUS_inject_latent_vector.png" style="width: 80%;">
</div>

</bar>
2. $\beta$-schedule

- train AE($\beta$=0) for 0.5 proportion
- anneal $\beta$ from 0 to 1 for 0.25 proportion
- fix $\beta$=1 for 0.25 proportion
- when $\beta$>0, $\mathcal{L_R}=\sum_i\max[\lambda, \text{KL}(q_\varPhi(z_i|x)||p(z_i))], where \lambda\in[0, 1]$

<div class="one-image-container">
    <img src="\AI\Paper_Reading\LM\Model_Category\VAE\image\OPTIMUS_beta_select.png" style="width: 80%;">
</div>

</bar>
3. OPTIMUS outperforms and adapts faster than BERT

<div class="one-image-container">
    <img src="\AI\Paper_Reading\LM\Model_Category\VAE\image\OPTIMUS_nlp_understanding.png" style="width: 80%;">

    <img src="\AI\Paper_Reading\LM\Model_Category\VAE\image\OPTIMUS_adapt_curve.png" style="width: 70%;">
</div>

</bar>
4. OPTIMUS learns a smoother and more structured latent space

<div class="one-image-container">
    <img src="\AI\Paper_Reading\LM\Model_Category\VAE\image\OPTIMUS_latent_space_tnse.png" style="width: 80%;">
</div>

</bar>
5. sentence transfer and interpolation abilities

<div class="one-image-container">
    <img src="\AI\Paper_Reading\LM\Model_Category\VAE\image\OPTIMUS_sentence_transfer.png" style="width: 80%;">
</div>
OPTIMUS在句子风格迁移改写任务中效果表现优越

<div class="one-image-container">
    <img src="\AI\Paper_Reading\LM\Model_Category\VAE\image\OPTIMUS_interpolation.png" style="width: 80%;">
</div>
OPTIMUS在内容主体单复数、长短句、相似语言等交织效果表现出平滑的效果

### BN-VAE
> 论文：[A Batch Normalized Inference Network Keeps the KL Vanishing Away](pdf/BN-VAE.pdf)  
> Tencent AI, ACL, 2020

#### 模型架构
<div class="one-image-container">
    <img src="\AI\Paper_Reading\LM\Model_Category\VAE\image\BN-VAE_algorithm.png" style="width: 40%;">
    <p style="text-align: center;">pipeline: encoder + VAE_distribution + BN + VAE_sample + decoder</p>
</div>



#### 工作要点
1. BN-VAE guarantee a positive lower bound of E[KL]
<div class="one-image-container">
    <img src="\AI\Paper_Reading\LM\Model_Category\VAE\image\BN-VAE_KL.png" style="width: 40%;">

    </br>
    <img src="\AI\Paper_Reading\LM\Model_Category\VAE\image\BN-VAE_KL_expectation_lower_bound.png" style="width: 35%;">
</div>

最低下限受dimension $n$和均值$\mu$的影响，前者超参固定，后者可通过batch norm $\mu \in \mathbb{R}^n$ 进行约束【即使在NLP任务中，也不受(padded_)seq维度影响】
> $\gamma$ and $\beta$ in batch norm are fixed

</br>
2. BN-VAE possesses high accuracy and coverage speed
<div class="row-image-container">
    <div>
        <img src="\AI\Paper_Reading\LM\Model_Category\VAE\image\BN-VAE_covergence_speed.png" style="width: 100%;">
        <!-- <p></p> -->
    </div>
    <div>
        <img src="\AI\Paper_Reading\LM\Model_Category\VAE\image\BN-VAE_accuracy_varied_from_samples.png" style="width: 80%;">
        <!-- <p>spatial dropout</p> -->
    </div>
</div>

</br>
3. the model posterior $p_\theta(z|x)$ is well learned with the help of the BN-VAE
decoder.
<div class="one-image-container">
    <img src="\AI\Paper_Reading\LM\Model_Category\VAE\image\BN-VAE_VAE_posterior_mean_comparion.png" style="width: 90%;">
</div>

### CEVE
> 论文：[Contrastive Deterministic Autoencoders For Language Modeling](pdf/CEVE.pdf)

#### 模型架构
<div class="one-image-container">
    <img src="\AI\Paper_Reading\LM\Model_Category\VAE\image\CEVE_framework.png" style="width: 60%;">
</div>

基于VAE每次采样结果不同的特性，CEVE进一步应用了对比学习的思想。