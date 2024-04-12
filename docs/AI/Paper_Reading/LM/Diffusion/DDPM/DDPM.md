### 前向加噪过程

#### 前向扩散
<div class="one-image-container">
    <img src="\AI\Paper_Reading\LM\Diffusion\DDPM\images\diffusion前向扩散.png" style="width: 80%;">
    <!-- <p style="text-align: center;">图片标题</p> -->
</div>

表示通过对一个图片不断地增加（高斯）噪声的权重，在足够的time_step后将获得趋近于一个噪声的图片，结果满足以下公式：

$$
\begin{aligned}
    x_{t} &= \sqrt{\alpha_t}x_{t-1=} + \sqrt{1-\alpha_t}\epsilon \\
    &= \sqrt{\bar\alpha_t}x_0 + \sqrt{1-\bar\alpha_t}\epsilon
\end{aligned}
$$



!!! info ""
    每一步加入的高斯噪声互相独立，因此正态分布可以进行叠加

其中$\beta_t$表示t时刻加入的噪声权重，$\alpha_t=1-\beta_t$表示图像$x_t$中上一时刻图像$x_{t-1}$的权重，$\bar\alpha_t=\prod_{i=1}^{t-1}\alpha_i$，即

<div class="one-image-container">
    <img src="\AI\Paper_Reading\LM\Diffusion\DDPM\images\diffusion前向扩散建模.png" style="width: 80%;">
    <p style="text-align: center;"><a href="https://wangjia184.github.io/diffusion_model/#">前向扩散示意图</a></p>
</div>

#### train model

1. 随机选定某一时刻$t\in [1, T]$，对原始图片 $x_0$ 加入噪声 $\epsilon \in \mathcal{N}(0, I)$ 生成 $x_t$
2. 将 $time\_embedding_t$ （类似于`position_embedding`）以及 $x_t$ 输入模型（e.g., `UNet`、Transformer）中，生成对加入噪声的预测结果 $\hat\epsilon$
3. 计算两个正态分布的 $D_{KL}(\mathcal{N}\Vert \mathcal{N}_\theta)$ 作为目标函数 $loss$ 以使模型拟合先验假设 {>>通过对一个图片不断地增加（高斯）噪声的权重，在足够的time_step后将获得趋近于一个噪声的图片<<}

$$
\begin{aligned}
    D_{KL}(\mathcal{N}_1\Vert \mathcal{N}_2) &= \log\frac{\sigma_2}{\sigma_1} + \frac{\sigma_1^2+(\mu_1-\mu_2)^2}{2\sigma_2^2} - \frac{1}{2} \\
    \mathcal{L} &= E_{q(x_0:T)}\Big[ \frac{1}{2\Vert \Sigma_\theta(x_t, t)\Vert^2}\Vert \mu(x_t, t) - \mu_\theta(x_t, t) \Vert^2\Big] + C \\
    &= E_{q(x_0:T), \epsilon\in\mathcal{N}(0, I)}\Big[ \frac{1}{2\Vert \Sigma_\theta(x_t, t)\Vert^2}\Vert \frac{1}{\sqrt{\alpha_t}}(x_t - \frac{\beta_t}{1-\bar\alpha_t}\epsilon) - \frac{1}{\sqrt{\alpha_t}}(x_t - \frac{\beta_t}{1-\bar\alpha_t}\epsilon_\theta(x_t,t)) \Vert^2\Big] + C \\
    \mathcal{L}_{simple} &= E_{q(x_0:T), \epsilon\in\mathcal{N}(0, I)}\Big[\Vert \epsilon - \epsilon_\theta(\sqrt{\bar\alpha_t}x_0 + \sqrt{1-\bar\alpha_t\epsilon}), t)\Vert^2\Big]
\end{aligned}
$$

<div class="admonition info" style="margin-left: 20px;">
    <p>最终loss为加入噪声 $\epsilon$ 和预测的噪声 $\hat\epsilon$ 间的MSE</p>
</div>  

<div class="one-image-container">
    <img src="\AI\Paper_Reading\LM\Diffusion\DDPM\images\ddpm_train_algorithm.jpg" style="width: 50%;">
    <!-- <p style="text-align: center;"><a href="https://wangjia184.github.io/diffusion_model/#">前向扩散示意图</a></p> -->
</div>


### DDPM逆向去噪过程

#### 逆向扩散
<div class="one-image-container">
    <img src="\AI\Paper_Reading\LM\Diffusion\DDPM\images\diffusion逆向扩散.png" style="width: 80%;">
    <!-- <p style="text-align: center;">图片标题</p> -->
</div>

DDPM (**D**enoising **D**iffusion **P**robabilistic **M**odel)，认为目标图片可以通过对噪声图片逐步地进行去噪操作实现图片的恢复重现，实现的数学理论如下：

$$
\begin{aligned}
    q(x_{t-1}|x_t, x_0)&=q(x_t|x_{t-1}, x_0)\frac{q(x_{t-1}|x_0)}{q(x_t|x_0)} &&贝叶斯公式 \\
    q(x_t|x_{t-1}, x_0)&=\sqrt{\alpha_t}x_{t-1} + \sqrt{1-\alpha_t}\epsilon &&\sim\mathcal{N}(\sqrt{\alpha_t}x_{t-1}, 1-\alpha_t I) \\
    q(x_{t-1}|x_0)&=\sqrt{\bar\alpha_{t-1}}x_0 + \sqrt{1-\bar\alpha_{t-1}}\epsilon &&\sim\mathcal{N}(\sqrt{\bar\alpha_{t-1}}x_0, 1-\bar\alpha_{t-1} I) \\
    q(x_{t}|x_0)&=\sqrt{\bar\alpha_{t}}x_0 + \sqrt{1-\bar\alpha_{t}}\epsilon&&\sim\mathcal{N}(\sqrt{\bar\alpha_{t}}x_0, 1-\bar\alpha_{t} I)
\end{aligned}
$$


若随机变量 $x$ 服从一个位置参数为 $\mu$、尺度参数为 $\sigma$ 的概率分布，且其概率密度函数为$f(x)=\frac{1}{\sqrt{2\pi}\sigma}\exp(-\frac{(x-\mu)^2}{2\sigma^2})$，因此上式可表示为：

$$
\begin{aligned}
    q(x_{t-1}|x_t, x_0)&=\exp\bigg(-\frac{1}{2}(\Big(\frac{\alpha_t}{\beta_t}+\frac{1}{1-\bar\alpha_{t-1}}\Big)x_{t-1}^2 - 2\Big(\frac{\sqrt{\alpha_t}}{\beta_t}x_t + \frac{\sqrt{\bar\alpha_{t-1}}}{1-\bar\alpha_{t-1}}x_0\Big)x_{t-1} + C(x_t, x_0))\bigg) \\
    \frac{1}{\sigma_{t-1}^2}&= \frac{\alpha_t}{\beta_t}+\frac{1}{1-\bar\alpha_{t-1}} \\
    \sigma_{t-1} &= \sqrt{\frac{1-\bar\alpha_{t-1}}{1-\bar\alpha_t}\beta_t} \\
    \mu_{t-1} &= \frac{\sqrt{\alpha_t}(1-\bar\alpha_{t-1})}{1-\bar\alpha_t}x_t + \frac{\sqrt{\bar\alpha_{t-1}}\beta_t}{1-\bar\alpha_t}x_0 \\
    &= \frac{1}{\sqrt{\alpha_t}}(x_t - \frac{\beta_t}{1-\bar\alpha_t}\epsilon)
\end{aligned} \\
$$



#### sample image

1. 初始化 $x_T\sim\mathcal{N}\text{(}0, \text{I}\text{)}$
2. 通过 $x_{t}$ 和 $time\_embedding_t$ 预测增加的噪声 $\hat\epsilon \sim \mathcal{N} \text{(}\mu_{t-1}, \sigma_{t-1}^2\text{)}$
3. ==基于 $\hat\epsilon$ 得到的分布采样 $t-1$ 时刻的图像$x_{t-1}$==
4. 重复2-3步直至得到$x_0$

<div class="admonition info" style="margin-left: 20px;">
    <p>由于每个time_step的图像都是采样得到的，因此diffusion模型具有很好的多样性表现</p>
</div>  


<div class="one-image-container">
    <img src="\AI\Paper_Reading\LM\Diffusion\DDPM\images\ddpm_sample_algorithm.jpg" style="width: 50%;">
    <!-- <p style="text-align: center;"><a href="https://wangjia184.github.io/diffusion_model/#">前向扩散示意图</a></p> -->
</div>