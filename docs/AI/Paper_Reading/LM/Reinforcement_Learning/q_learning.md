## Q-Learning
Q-Learning 是强化学习（Reinforcement Learning, RL）中最经典的**无模型**（model-free）算法之一，由 Chris Watkins 在 1989 年提出，适用于离散状态和动作空间的问题。

### 主要内容
Q-Learning 通过直接优化更新动作价值函数或状态价值函数，选择最优动作，找到最佳策略

$$
\pi(s) =\text{arg} \mathop{\text{ max }}\limits_{a} Q(s, a)
$$

#### Bellman Equation
贝尔曼方程的核心思想是：`当前状态的价值=即时奖励 + 未来状态的折扣价值`。根据不同的价值函数，分为以下两种形式：

1. **动作价值函数** $Q$ 的贝尔曼方程，

    $$
    \begin{aligned}
        Q^{\pi}(s_t, a_t) 
        =& \mathbb{E}_{\pi} [r_{t+1} + \gamma Q^{\pi}(s_{t+1}, a_{t+1})\vert s_t, a_t] 
    \end{aligned}
    $$

    > - $\gamma \in [0, 1]$ 为折扣因子，用于平衡当前与未来奖励的重要性（时序越远影响越小）  
    > - $\mathbb{E}$ 表示对所有概率下状态期望化，即 $\sum_{s_{t+1}^{'}} p(s_{t+1}^{'}\vert s_t, a_t)$

2. **状态价值函数** $V$ 的贝尔曼方程

    $$
    V^{\pi}(s_t) = \mathbb{E}_{\pi} [r_{t+1} + \gamma V^{\pi}(s_{t+1})\vert s_t]
    $$

    > - $\mathbb{E}$ 表示对所有动作概率下状态期望化，即 $\sum_{a_{t}^{'}} p_{\pi}(a_{t}^{'}\vert s_t)\sum_{s_{t+1}^{'}}p(s_{t+1}^{'}\vert s_t, a_{t}^{'})$

#### 策略求取步骤
通过递归求解 $Q$ 或 $V$ 即可得到最优策略 $\pi^*$

## DQN


## Double DQN

## Dueling DQN