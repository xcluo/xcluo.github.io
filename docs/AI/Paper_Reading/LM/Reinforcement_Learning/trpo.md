## TRPO
> 论文：**T**rust **R**egion **P**olicy **O**ptimization  
> University of California, 2015 Feq, PMLR 2015

### 主要内容
TRPO只在置信区间内进行策略优化，即通过clip操作约束策略优化区间

$$
\max_{\theta} \hat{\mathbb{E}}_t\bigg[  \frac{\pi_{\theta}(a_t\vert s_t)}{\pi_{old}(a_t\vert s_t)} \hat{A}_t\bigg] \\
\text{subject to } \hat{\mathbb{E}}_t[D_{KL}\big(\pi_{old}(\cdot \vert s_t), \pi_{\theta}(\cdot \vert s_t)\big)] \le \delta
$$

> 在SGA的同时要求遵循两个模型的策略分布差距处于较高的相似度的硬约束（$\le \delta$）

$$
\max_{\theta} \hat{\mathbb{E}}_t\bigg[  \frac{\pi_{\theta}(a_t\vert s_t)}{\pi_{old}(a_t\vert s_t)} \hat{A}_t -\beta D_{KL}\big(\pi_{old}(\cdot \vert s_t), \pi_{\theta}(\cdot \vert s_t)\big)\bigg] \\
$$

> 整合为惩罚项，SGA - D_\{KL\}