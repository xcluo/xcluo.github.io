## YaRN
> 论文：YaRN: Efficient ContextWindow Extension of Large Language Models  
> YaRN: **Y**et **a**nother **R**oPE extensio**N** method  
> Github: [yarn](https://github.com/jquesnelle/yarn)  
> Nous Research & EleutherAI & University of Geneva, 2023 Aug, ICLR 2024

### 主要内容
- YARN = NTK-by-parts + attention scaling
- $f^{'}(x,m, \theta_i)$
- 对Attention分母乘以了$t$，可理解为对$q$和$k$均进行温度因子 $\frac{1}{\sqrt{t}} = 1 + 0.1 \ln s$ 的缩放，$s = \max(1, l/L_{train})$
- 因为RoPE具有长度衰减特性，当插值后相对距离会减小（插值后$/s$值会变大，相关性增大），因此需要通过乘以 $\frac{1}{\sqrt{t}}$进行平滑还原缩放
- 不改变向量模长，但是改变向量夹角，即 $q^Tk\approx d\cos(q, k)$
- 在微调和非微调场景下均超过以往所有方法
- 只需要对不到原始预训练数据的0.1%进行微调，YaRN的上下窗口拓展到最先进的性能

YaRN通过三个关键机制解决位置外推问题

1. dynamic-NTK 插值，动态调整RoPE基频，平衡高频（$l \gt L_{train}$）、低频信息（$l \lt L_{train}$），符合动态调整特性  
2. 动态温度缩放，自适应调整注意力分数logits， $\frac{1}{\sqrt{t}} = 1 +\frac{\log \alpha}{d} \ln s$
3. 衰减补偿，保护局部注意力模式，即对低维高频部分逐渐开放，使得随维度增加，逐步抑制高频，防止由于旋转过快导致相邻位置编码相似度过小

    $$
    \begin{aligned}
        \lambda_i =& 1 - \gamma \frac{i}{d/2}, 0 \le i \lt d/2 \\
        \theta_i^{'} =& \lambda_i\cdot \theta_i
    \end{aligned}
    $$

    > 建议取值，小模型`d <= 2556`，γ=0.05~0.1；大模型`d > 256`，γ=0.1~0.15

- [x] Attention softmax后除以$\sqrt{d_h}$是因为权重矩阵中每个元素都是通过两个(d_h, 1)各维度为独立同分布的均值为0方差为1的向量相乘得到的，基于正态分布累加后的标准差公式可知该值方差变为$\sqrt{d_h}$，因此执行该操作，若不除以$\sqrt{d_h}$，根据softmax函数曲线，softmax结果表现更倾向于one-hot分布，[会带来梯度消失问题](https://spaces.ac.cn/archives/8620/comment-page-4#comment-24076)

- truncated normal的基于正态分布 $\mathcal{N}(\mu, \sigma^2)$，对于在$[u-2\sigma, u+2\sigma]$范围内采样结果保留，其均值为$\mu$，方差为

    $$
    \gamma = \frac{\int_{-2}^2 e^{-x^2/2}x^2 dx}{\int_{-2}^2 e^{-x^2/2} dx} = 0.7737413
    $$

- 若要得到方差为$\sigma^2$ 采样结果，需要对传入的标准差执行 $\sigma *= \frac{1}{\sqrt{\gamma}} = 1.1368472\sigma$
- https://spaces.ac.cn/archives/8620
- https://spaces.ac.cn/archives/8823
- https://spaces.ac.cn/archives/9948
- https://spaces.ac.cn/archives/9859

Attention机制

$$
\begin{aligned}
    o_i = \sum_{j=1}^n a_{i, j} v_j \\
    a_{i,j} = \frac{e^{\lambda q_i^T k_j}}{\sum_{j=1}^n e^{\lambda q_i^T k_j}}
\end{aligned}
$$

- 正常情况下 $\lambda = \frac{1}{\sqrt{d_h}}$
- 为了使得模型结果能够更好地泛化到更大长度，Attention机制在拓展时应该使得$a_{i,j}$ 尽量具备熵不变性（对长度n不敏感），即 $H_i = -\sum a_{i, j} \log a_{i, j}$
- 所有注意力都集中在一个token熵 $H_i = 0$，注意力均匀分布在所有token上 $H_i = -\frac{1}{n}\log \frac{1}{n} = \log n$
- 进一步拆解$H_i$

$$
\begin{aligned}
    H_i =& -\sum_{j=1}^n a_{i, j} \log a_{i, j} \\
    =& -\sum_{j=1}^n  \frac{e^{\lambda q_i^T k_j}}{\sum_{j=1}^n e^{\lambda q_i^T k_j}} \log \frac{e^{\lambda q_i^T k_j}}{\sum_{j=1}^n e^{\lambda q_i^T k_j}} \\
    =& \frac{\sum 子 \log 母}{母} - \frac{\sum 子 \log 子}{母}= \log 母 - \frac{\sum 子 \log 子}{母} \\
    =& \log \sum_{j=1}^n e^{\lambda q_i^T k_j} - \frac{\sum_{j=1}^n e^{\lambda q_i^T k_j}(\lambda q_i^T k_j)}{\sum_{j=1}^n e^{\lambda q_i^T k_j}}
\end{aligned}
$$

- 对于 $\sum_{j=1}^n e^{\lambda q_i^T k_j} = n\times \frac{1}{n} \sum_{j=1}^n e^{\lambda q_i^T k_j} \approx n \mathbb{E}_j[e^{\lambda q_i^T k_j}]$

$$
H_i \approx \log n \mathbb{E}_j[e^{\lambda q_i^T k_j}] - \frac{\lambda \mathbb{E}_j[e^{\lambda q_i^T k_j}(q_i^Tk_j)]}{\mathbb{E}_j[e^{\lambda q_i^T k_j}]}
$$

- 一般情况下，$q_i, k_j$ 都是经过LN + Dense层得来，而后者接近正交变换，因此可以近似假设 $q_i, k_j$ 都是模长为 $\sqrt{d_h}$ 的向量，所以 $q_i^Tk_j = d\cos(q_i, k_j)$，假设$k_j$均匀分布在向量空间内，那么对$k_j$的期望可以转化为对 $q_i, k_j$ 夹角余弦值的期望

$$
H_i \approx \log n + \log \mathbb{E}_{\theta} e^{\lambda d\cos \theta} - \frac{\lambda d \mathbb{E}_{\theta}[e^{\lambda d\cos \theta}\cos \theta]}{\mathbb{E}_{\theta}[e^{\lambda d\cos \theta}]}
$$

拉普拉斯近似可得, https://spaces.ac.cn/archives/7695#%E7%BB%93%E6%9E%9C%E5%AF%B9%E6%AF%94

$$
H_i \approx \log n - \frac{1}{2}\ln \frac{\sqrt{5} + 1}{2} = \log n -0.24\lambda d
$$

- 为了抵消长度$n$的影响，可以让 $\log n - 0.24\lambda d  =0$，即$\lambda \approx \frac{\log n}{0.24 d} =  \frac{\log s + \log L}{0.24 d}$，可以引申为 $\lambda = \frac{k}{d} \log n$  
- 此时可表示为 $\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{k \log n}{d} QK^T\right)$，对于原始长度$L$，即 $\frac{k\log L}{d} = \frac{1}{\sqrt{d}}$，$k=\frac{\sqrt{d}}{\log L}$，重新带入公式得此$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{\log_L n}{\sqrt{d}} QK^T\right)$，对于context window拓展$L'=sL$，$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{\log_L sL'}{\sqrt{d}} QK^T\right) = \left(\frac{\log_L s + 1}{\sqrt{d}} QK^T\right)$


- yarn中 $\frac{1}{t}=(0.1\log s + 1)^2 \approx 0.2\log s + 1$ 
- 推导值 $\log_L s +1 = \frac{\ln s}{\ln L} + 1$
- 当L为2K时，推导值$\approx 0.13\ln s + 1$；当L为4K时，推导值$\approx 0.12\ln s + 1$