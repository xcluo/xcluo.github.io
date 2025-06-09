## DAPO

> 论文：DAPO: An Open-Source LLM Reinforcement Learning System at Scale  
> Project Page：[dapo-sia.github.io](https://dapo-sia.github.io/)  
> DAPO：Decoupled clip **D**ynamic s**A**mpling **P**olicy **O**ptimization  
> ByteDance Seed & Tsinghua University & The University of Hong Kong, 2025 Mar

### 主要内容
- https://mp.weixin.qq.com/s?__biz=MzI4MDYzNzg4Mw==&mid=2247571337&idx=2&sn=ca7ed117c5f5534bc4299ca5f166f9f0&chksm=ea4791a30d92493bfc6f54ff44bcc1d4c6f5185b73dc1f94d804baeac25f909978fd2facb516&scene=27
- Qwen2.5-32B as the pretrained model for RL
- GRPO baseline suffers from several key issues such as 1) entropy collapse; 2) reward noise; 3) training instability
- DAPO introduces 4 key techniques to make RL shine in long-CoT RL schenario
    1. clip-higher
    2. dynamic sampling
    3. token-level policy gradient loss
    4. overlong reward shaping

- PPO in LLM 

    $\mathcal{L}_\text{PPO} = \mathbb{E}_{(q, a) \sim D, o\le t \sim \pi_{\theta_\text{old}}(\cdot \vert q)} \bigg[ \min \Big(
        \frac{\pi_{\theta}(o_t\vert q, o\lt t)}{\pi_{\theta_\text{old}}(o_t\vert q, o\lt t)}\hat{A}_t, 
        \text{clip}\big(
            \frac{\pi_{\theta_\text{old}}(o_t\vert q, o\lt t)}{\pi_{\theta_\text{old}}(o_t\vert q, o\lt t)}, 1-\epsilon, 1+\epsilon
            \big)\hat{A_t}\Big) \bigg]$

    > $(q, a)$ 为question-answer pair  
    > $o$ 表示reference model的output


- GRPO in LLM

    $\begin{aligned}
        \hat{A}_{i, t} =& \frac{r_i - \text{avg}(\{r_j\}_{j=1}^G)}{\text{std}(\{r_j\}_{j=1}^G)} \\
        \mathcal{L}_\text{GRPO} = \mathbb{E}_{(q, a) \sim D, o\le t \sim \pi_{\theta_\text{old}}(\cdot \vert q)} \bigg[ \min \Big(&
        \frac{\pi_{\theta}(o_t\vert q, o\lt t)}{\pi_{\theta_\text{old}}(o_t\vert q, o\lt t)}\hat{A}_t, 
        \text{clip}\big(
            \frac{\pi_{\theta_\text{old}}(o_t\vert q, o\lt t)}{\pi_{\theta_\text{old}}(o_t\vert q, o\lt t)}, 1-\epsilon, 1+\epsilon
            \big)\hat{A_t}\Big) \bigg]
    \end{aligned}$