### 前向加噪过程

#### 前向扩散
<div class="one-image-container">
    <img src="\AI\Paper_Reading\LM\Diffusion\DDPM\images\diffusion前向扩散.png" style="width: 80%;">
    <!-- <p style="text-align: center;">图片标题</p> -->
</div>

通过对对一个图片不断地增加（高斯）噪声的权重，在足够的time_step后将获得趋近于一个噪声的图片，结果满足以下公式：

$$
\begin{aligned}
    x_{t} &= \sqrt{\alpha_t}x_{t-1=} + \sqrt{1-\alpha_t}\epsilon \\
    &= \sqrt{\bar\alpha_t}x_0 + \sqrt{1-\bar\alpha_t}\epsilon
\end{aligned}
$$

其中$\beta_t$表示t时刻加入的噪声权重，$\alpha_t=1-\beta_t$表示图像$x_t$中上一时刻图像$x_{t-1}$的权重，$\bar\alpha_t=\prod_{i=1}^{t-1}\alpha_i$，即

<div class="one-image-container">
    <img src="\AI\Paper_Reading\LM\Diffusion\DDPM\images\diffusion前向扩散建模.png" style="width: 80%;">
    <p style="text-align: center;"><a href="https://wangjia184.github.io/diffusion_model/#">前向扩散示意图</a></p>
</div>

#### train model

1. 随机选定某一时刻$t\in [1, T]$，对原始图片 $x_0$ 加入噪声 $\epsilon$ 生成 $x_t$
2. 将 $time\_embedding_t$ （类似于`position_embedding`）以及 $x_t$ 输入模型（e.g., `UNet`、Transformer）中，生成对加入噪声的预测结果 $\hat\epsilon$
3. 计算两个正态分布的 $D_{KL}(\epsilon|\hat\epsilon)$ 作为目标函数 $loss$ 以使模型拟合先验假设 {>>通过对一个图片不断地增加（高斯）噪声的权重，在足够的time_step后将获得趋近于一个噪声的图片<<}


### 逆向去噪过程

#### 逆向扩散
<div class="one-image-container">
    <img src="\AI\Paper_Reading\LM\Diffusion\DDPM\images\diffusion逆向扩散.png" style="width: 80%;">
    <!-- <p style="text-align: center;">图片标题</p> -->
</div>

$$
\begin{aligned}
    q(x_{t-1}|x_t, x_0)&=q(x_t|x_{t-1}, x_0)\frac{q(x_{t-1}|x_0)}{q(x_t|x_0)}，贝叶斯公式 \\
    q(x_t|x_{t-1}, x_0)&=\sqrt{\alpha_t}x_{t-1} + \sqrt{1-\alpha_t}\epsilon & \sim\mathcal{N}(\sqrt{\alpha_t}x_{t-1}, 1-\alpha_t) \\
    q(x_{t-1}|x_0)&=\sqrt{\bar\alpha_{t-1}}x_0 + \sqrt{1-\bar\alpha_{t-1}}\epsilon &\sim\mathcal{N}(\sqrt{\bar\alpha_{t-1}}x_0, 1-\bar\alpha_{t-1}) \\
    q(x_{t}|x_0)&=\sqrt{\bar\alpha_{t}}x_0 + \sqrt{1-\bar\alpha_{t}}\epsilon &\sim\mathcal{N}(\sqrt{\bar\alpha_{t}}x_0, 1-\bar\alpha_{t})
\end{aligned}
$$


#### generate image

1. 初始化noise为 $x_T$
2. 通过$x_{t}$预测增加的噪声 $\hat\epsilon$
3. 基于 $\hat\epsilon$ 预测$t-1$ 时刻图像分布 ($u_{t-1}$, $\sigma_{t-1}^2$)，并采样得到$x_{t-1}$，重复1-2步直至$x_0$