### BN
即 Batch Normalization




### LN
即 Layer Normalization

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
即 Root Mean Squared Layer Normalization，==RMS认为LN取得的成功是缩放不变性，而不是平移不变性，因此较LN只保留了缩放（除以标准差）==，去除了平移（减去均值）

$$\text{RMS}(x_i)=\frac{x_i}{\sqrt{\frac{1}{d}\sum_{1}^{d}x_i^2}+\epsilon}$$