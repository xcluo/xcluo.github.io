## PG
策略梯度 Policy Gradient 是强化学习中一类直接优化策略的方法，通过梯度上升直接调整策略参数，适用于连续动作空间和随机策略的场景。

### 主要内容
PG 的核心思想为最大化轨迹（折扣或非折扣）回报

$$
J(\theta) = \mathbb{E}_{\tau \text{\textasciitilde} \pi_{\theta}} \bigg[ \sum_{t=0}^\infty \gamma^{t}r_t \bigg]
$$

#### 梯度推导

$$
\begin{aligned}
    \nabla_{\theta} J(\theta) =& \mathbb{E}_{\tau \text{\textasciitilde} \pi_{\theta}} \bigg[ \sum_{t=0}^{T_n} \Psi_t \nabla_{\theta} \log \pi_{\theta}(a_t\vert s_t) \bigg]

\end{aligned}
$$

- 每一步都进行了上述优化

$\Psi_t$ 有以下多种选择：

- $Q^{\pi}(s_t, a_t)$，动作价值函数
- $A^{\pi}(s_t, a_t)$，优势函数，→ 最小方差
- $\sum_{t=0}^{\infty} r_t$，轨迹（折扣或非折扣）总回报
- $\sum_{k=0}^{\infty} r_{t+k}$，后续轨迹（折扣或非折扣）回报
- $\sum_{k=0}^{\infty} r_{t+k} - b_{t+k}$，减去基线的后续轨迹（折扣或非折扣）回报
- $r_t + V^{\pi}(s_{t+1}) - V^{\pi}(s_t)$，时序差分残差Temporal Difference residual，前两项可理解为动作价值矩阵 $Q^{\pi}(s_t, a_t) = r_t + V^{\pi}(s_{t+1})$，只不过前者只需要一个模型，后者需要两个模型，优于采用了动作也会有各种状态转移结果，此时需要对所有$V(s_{t+1})$的取均值，即$\mathbb{E}_{s_{t+1}}[r_t + \gamma V^{\pi, \gamma}(s_{t+1}) - V^{\pi, \gamma}(s_t)]$


1. 蒙特卡洛（通过采样估计）策略梯度

    $$
    \nabla_{\theta} J(\theta) \approx \frac{1}{N}\sum_{i=1}^{N}\sum_{t=0}^{T_n} \nabla_{\theta} \log \pi_{\theta}(a_t^i\vert s_t^i)\cdot G_{t}^i 
    $$

    > 蒙特卡洛回报 $G_{t} = \sum_{t^{'}=t}^{T_n} \gamma^{t^{'} - t}\cdot r_{t}$

2. Actor-Critic

    $$
    \nabla_{\theta} J(\theta) \approxcolon \mathbb{E} \big[  \nabla_{\theta} \log \pi_{\theta}(a_t\vert s_t)\cdot A^{\pi_\theta}(s_t, a_t) \big]
    $$

    > 通过估计每一步的优势值替代蒙特卡洛回报 $G_t$


- vanilla policy gradient with adaptive stepsize