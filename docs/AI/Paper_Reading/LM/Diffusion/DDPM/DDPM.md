#### 正向加噪过程（train训练）
$$
\begin{aligned}
    x_{t} &= \sqrt{\alpha_t}x_{t-1=} + \sqrt{1-\alpha_t}\epsilon \\
    &= \sqrt{\bar\alpha_t}x_0 + \sqrt{1-\bar\alpha_t}\epsilon
\end{aligned}
$$


1. 通过加噪 $\epsilon$ 生成 $x_t$
2. `UNet`通过 $x_t$ 图像预测增加的噪声 $\hat\epsilon$
3. 计算$\hat\epsilon$ 和 $\epsilon$的KL散度，即$\text{KL}(\hat\epsilon|\epsilon)$


#### 反向去噪过程（infer生成）


$$
\begin{aligned}
    q(x_{t-1}|x_t, x_0)&=q(x_t|x_{t-1}, x_0)\frac{q(x_{t-1}|x_0)}{q(x_t|x_0)}，贝叶斯公式 \\
    q(x_t|x_{t-1}, x_0)&=\sqrt{\alpha_t}x_{t-1} + \sqrt{1-\alpha_t}\epsilon & \sim\mathcal{N}(\sqrt{\alpha_t}x_{t-1}, 1-\alpha_t) \\
    q(x_{t-1}|x_0)&=\sqrt{\bar\alpha_{t-1}}x_0 + \sqrt{1-\bar\alpha_{t-1}}\epsilon &\sim\mathcal{N}(\sqrt{\bar\alpha_{t-1}}x_0, 1-\bar\alpha_{t-1}) \\
    q(x_{t}|x_0)&=\sqrt{\bar\alpha_{t}}x_0 + \sqrt{1-\bar\alpha_{t}}\epsilon &\sim\mathcal{N}(\sqrt{\bar\alpha_{t}}x_0, 1-\bar\alpha_{t})
\end{aligned}
$$



1. 初始化noise为 $x_T$
2. 通过$x_{t}$预测增加的噪声 $\hat\epsilon$
3. 基于 $\hat\epsilon$ 预测$t-1$ 时刻图像分布 ($u_{t-1}$, $\sigma_{t-1}^2$)，并采样得到$x_{t-1}$，重复1-2步直至$x_0$