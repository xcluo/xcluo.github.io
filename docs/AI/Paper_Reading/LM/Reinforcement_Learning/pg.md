## PG
策略梯度 Policy Gradient 是强化学习中一类直接优化策略的方法，通过梯度上升直接调整策略参数，适用于连续动作空间和随机策略的场景。

### 主要内容
PG 的核心思想为最大化轨迹回报

$$
J(\theta) = \mathbb{E}_{\tau \text{\textasciitilde} \pi_{\theta}} \bigg[ \sum_{t=0}^\infty \gamma^{t}r_t \bigg]
$$

#### 梯度推导

$$
\begin{aligned}
    \nabla_{\theta} J(\theta) =& \mathbb{E}_{\tau \text{\textasciitilde} \pi_{\theta}} \bigg[ \sum_{t=0}^{T_n} \nabla_{\theta} \log \pi_{\theta}(a_t\vert s_t)\cdot Q^{\pi_{\theta}}(s_t, a_t) \bigg]

\end{aligned}
$$

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