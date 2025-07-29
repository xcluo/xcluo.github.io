- RMSNorm只有缩放参数
- LN额外增加偏移参数
- BN在上述基础上动量保留历史方差和均值

### BN
即 Batch Normalization



### LN
即 Layer Normalization，基于样本所有维度信息进行平移不换转化和缩放不变转化，随后进行投影变换。

$$
\begin{aligned}
    \text{LN}(x_i) = \hat{x_i} =&\frac{x_i - \mu}{\sqrt{\frac{1}{d}\sum_{j=1}^d (x_j - \mu)^2} + \epsilon}    \\
    y_i =& \gamma_i\hat{x_i} + \beta_i \\
    &\gamma, \beta \in \mathbb{R}^{d}
\end{aligned}
$$

#### Pre-Norm & Post-Norm
[Pre-Norm/Post-Norm](pre-norm_post-norm.md) 是指在残差连接操作之前/之后执行Norm操作，直白的区别是在要用的时候进行Norm操作还是用之前就执行Norm操作

$$
\begin{aligned}
    x_{t+1}^{pre-norm} = x_t + F_t\big(\text{Norm}(x_t)\big) \\
    x_{t+1}^{post-norm} = \text{Norm}\big(x_t + F_t(x_t)\big)
\end{aligned}
$$

1. **Pre-Norm**
    - [x] 训练稳定，收敛快，适合深层模型
    - [x] 运用$h_t$作为预测层输入前最好进行Norm操作以归一化方差
    - 相同条件下较Post-Norm最优表达能力可能会略低。

2. **Post-Norm**
    - [x] 表达能力更强，但训练不稳定，收敛慢，适合浅层模型。
    - Post-Norm模型的训练极度依赖warmup

#### RMSNorm
即 Root Mean Squared Layer Normalization，==RMS认为LN取得的成功是缩放不变性，而不是平移不变性，因此较LN只保留了缩放转化（除以标准差）==，去除了平移转化（减去均值），随后进行无偏置项的投影变换

$$
\begin{aligned}
    \text{RMS}(x_i)=\hat{x_i} =& \frac{x_i}{\sqrt{\frac{1}{d}\sum_{j=1}^{d}x_j^2}+\epsilon} \\
    y_i =& \gamma_i \hat{x_i} \\
    & \gamma \in \mathbb{R}^d
\end{aligned}
$$