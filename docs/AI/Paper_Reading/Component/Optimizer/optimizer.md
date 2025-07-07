- An overview of gradient descent optimization algorithms

## Gradient Descent Variant

### SGD
随机梯度下降 Stochastic Gradient Descent，一次迭代使用单个样本或小批量(mini-batch)样本

$$
\begin{aligned}
    g_t =& \nabla_{\theta_{t-1}} J\left(\theta_{t-1}; x^{(i:i+n)}; y^{(i:i+n)}\right) \\
    \theta_t  =& \theta_{t-1} -\eta g_t    
\end{aligned}
$$

> $1 \le n \lt \text{batch_size}$

### BGD
批量梯度下降 Batch Gradient Descent, 一次迭代使用批量中所有样本

$$
\begin{aligned}
    g_t =& \nabla_{\theta_{t-1}} J\left(\theta_{t-1}; x; y\right) = \nabla_{\theta}J(\theta) \\
    \theta_t  =& \theta_{t-1} -\eta g_t    
\end{aligned}
$$

## Gradient Descent Optimization


### Momentum
动量法 Momentum 模拟物体运动时的惯性，在梯度更新时一定程度上保留之前更新的方向

$$
\begin{aligned}
    v_t =& \gamma v_{t-1} + \eta g_t \\
    \theta_t =& \theta_{t-1} - v_t
\end{aligned}
$$

#### NAG
Nesterov Accelerated Gradient，结合动量更新提前提前"跳跃"到一个前瞻位置，再在该位置计算梯度并修正，在凸优化问题中实现了更快的收敛速度

$$
\begin{aligned}
    v_t =& \gamma v_{t-1} + \eta \nabla_{\theta_{t-1}} J(\theta_{t-1} -\gamma v_{t-1}) \\
    \theta_t =& \theta_{t-1} - v_t
\end{aligned}
$$

### Adagrad
自适应学习率 $\eta$


$$
\begin{aligned}
    G_{t} =& \sqrt{\sum_{i=0}^{t}{(g_i)^2}} \\
    \theta_{t} =& \theta_{t-1}-\frac{\eta}{\sqrt{G_{t} + \epsilon}} \odot g_t\\
\end{aligned}
$$
#### Adadelta

$$
\begin{aligned}
    \mathbb{E}\left[ g^2 \right]_t =& \gamma \mathbb{E}\left[ g^2 \right]_{t-1} + (1 - \gamma) g^2_t \\
    \mathbb{E}\left[ \Delta \theta^2 \right]_t =& \gamma \mathbb{E}\left[ \Delta \theta^2 \right]_{t-1} + (1-\gamma) \Delta\theta^2_t \\
    \theta_{t} =& \theta_{t-1}-\frac{\sqrt{\mathbb{E}\left[ \Delta \theta^2 \right]_{t-1} + \epsilon}}{\sqrt{\mathbb{E}\left[ g^2 \right]_t + \epsilon}}\odot g_t\\
\end{aligned}
$$

> $\Delta \theta$ 表示模型参数更新量，即 $\Delta\theta_t = \theta_t - \theta_{t-1}$

#### RMSprop

$$
\begin{aligned}
    \mathbb{E}\left[ g^2 \right]_t =& \gamma \mathbb{E}\left[ g^2 \right]_{t-1} + (1 - \gamma) g^2_t \\
    \theta_{t} =& \theta_{t-1}-\frac{\eta}{\sqrt{\mathbb{E}\left[ g^2 \right]_t + \epsilon}}\odot g_t\\
\end{aligned}
$$

### Adam
Adaptive Moment Estimation

$$
\begin{aligned}
    m_t =& \beta_1 m_{t-1} + (1-\beta_1) g_t\\
    v_t =& \beta_2 v_{t-1} + (1-\beta_2) g_t^2\\
    \hat{m}_t =& \frac{m_t}{1-\beta_1^{t}} \\
    \hat{v}_t =& \frac{v_t}{1-\beta_2^{t}} \\
    \theta_{t} =& \theta_{t-1} -\frac{\eta}{\sqrt{\hat{v}_t} + \epsilon}\odot \hat{m}_t
\end{aligned}
$$

> $\hat{m}_t, \hat{v}_t$ 为初始阶段对 $m_t, v_t$ 的偏差纠正项，分母部分 $\lim\limits_{t \rightarrow \infty} \beta^t = 0$

#### AdaMax

$$
\begin{aligned}
    m_t =& \beta_1 m_{t-1} + (1-\beta_1) g_t \\
    v_t =& \max \left(\beta_2 v_{t-1}, \vert g_t \vert \right)\\
    \hat{m}_t =& \frac{m_t}{1-\beta_1^{t}} \\
    \theta_{t} =& \theta_{t-1} -\frac{\eta}{v_t}\odot \hat{m}_t
\end{aligned}
$$

> $v_t$ 使用无穷范数 $L_{\infty}$ 替代 $L_2$ 范数，并取消了偏差纠正项
#### Nadam
Nesterov-accelerated Adaptive Moment Estimation

$$
\begin{aligned}
    m_t =& \beta_1 m_{t-1} + (1-\beta_1) g_t\\
    v_t =& \beta_2 v_{t-1} + (1-\beta_2) g_t^2\\
    \hat{v}_t =& \frac{v_t}{1-\beta_2^{t}} \\
    \hat{m}_t^{'} =& \beta_1 m_{t} + \frac{1- \beta_1}{1-\beta^t_1}g_t\\
    \theta_{t} =& \theta_{t-1} -\frac{\eta}{\sqrt{\hat{v}_t} + \epsilon}\odot \hat{m}_t^{'}
\end{aligned}
$$

#### AdamW


$$
\begin{aligned}
    m_t =& \beta_1 m_{t-1} + (1-\beta_1) g_t\\
    v_t =& \beta_2 v_{t-1} + (1-\beta_2) g_t^2\\
    \hat{m}_t =& \frac{m_t}{1-\beta_1^{t}} \\
    \hat{v}_t =& \frac{v_t}{1-\beta_2^{t}} \\
    \theta_{t} =& \theta_{t-1} -\eta(\frac{\hat{m}_t}{{\sqrt{\hat{v}_t} + \epsilon}} + \lambda \theta_{t-1})
\end{aligned}
$$