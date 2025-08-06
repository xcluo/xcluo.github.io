## YaRN
> 论文：YaRN: Efficient ContextWindow Extension of Large Language Models  
> YaRN: **Y**et **a**nother **R**oPE extensio**N** method  
> Github: [yarn](https://github.com/jquesnelle/yarn)  
> Nous Research & EleutherAI & University of Geneva, 2023 Aug, ICLR 2024

### 主要内容
- YARN = NTK-by-parts + attention scaling
- $f^{'}(x,m, \theta_i)$
- 对Attention分母乘以了$t$，可理解为对$q$和$k$均进行温度因子 $\frac{1}{\sqrt{t}} = 1 +\frac{\log \alpha}{d} \ln s$ 的缩放，$s = \max(1, l/L_{train})$
- 因为RoPE具有长度衰减特性，当插值后较长部分值会减小，因此需要通过乘以 $\frac{1}{\sqrt{t}}$进行方法
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

