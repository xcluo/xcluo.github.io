## PPO
> 论文：**P**roximal **P**olicy **O**ptimization Algorithms  
> OpenAI 2017 Aug

----
- importance sampling，重要性采样
- with clipped probability ratios，防止和reference model差异过大，裁剪策略更新以至于重要性采样效果不对齐  
- 目标函数为优势函数 $A_t$ 

    $$
    \begin{aligned}
        L^{CLIP}(\theta) =& \mathbb{E}_t \bigg[\min \Big(\frac{\pi_{\theta}(a_t\vert s_t)}{\pi_{\theta_{old}}(a_t\vert s_t)}A_t, \text{clip}\big(\frac{\pi_{\theta}(a_t\vert s_t)}{\pi_{\theta_{old}}(a_t\vert s_t)}, 1-\epsilon, 1+\epsilon\big)A_t \Big) \bigg] \\
        =& \mathbb{E}_t \Big[\min \big(r_t(\theta) A_t, \text{clip}(r_t(\theta), 1-\epsilon, 1+\epsilon)A_t\big)\Big]
    \end{aligned}
    $$

    > 新旧策略概率比ratio  $r_t(\theta)=\frac{\pi_{\theta}(a_t\vert s_t)}{\pi_{\theta_{old}}(a_t\vert s_t)}$  
    > $\epsilon$ 为裁剪参数，通常取0.1 ~ 0.3，强制 $r_t(\theta)$ 接近1，避免过大更新

- GAE（Generalized Advantage Estimation），truncated version of generalized advantage estimation

    $$
    \begin{aligned}
        R_t =& \sum_{k=0}^{T-t} (\gamma \lambda)^k \delta_{t+k} \\
        \delta_t =& r_t + \gamma V(s_{t+1}) - V(s_t) \\
        V^{\pi}(s_t) =&\mathbb{E}_{\pi}\bigg[ \sum_{k=0}^{\infty} \gamma^{k}r_{t+k}\vert s_t\bigg]
    \end{aligned}
    $$

- 同时优化策略和值函数 $V$ （critic），值函数的目标是最小化均方差

    $$
    L^{VF}(\theta) = \mathbb{E}_{t} \Big[ \big(V_t^\text{targ} - V^{\pi}(s_t)\big)^2 \Big]
    $$

    - +clip：PPO-Clip  
    - +KL：PPO-Penalty

- put it all together

    $$
    L^{PPO}(\theta) = L^{CLIP}(\theta) - c_1L^{VF}(\theta) + c_2 H(\pi_\theta)
    $$

- entropy bonus $H$，$H = -\sum_{a} \pi_{\theta}(a|s_t)\log \pi_{\theta}(a\vert s_t)$
- discount $\gamma$  
- GAE parameter $\lambda$
- alternate between sampling data from the policy and performing several epochs of optimization on the sampled data，每批数据优化多轮（传统一批数据优化一轮）  
- PPO是on-policy算法，因为其训练依赖当前策略生成的数据。
- https://blog.csdn.net/shizheng_Li/article/details/144752966?sharetype=blogdetail&sharerId=144752966&sharerefer=PC&sharesource=shizheng_Li&spm=1011.2480.3001.8118
- GAE: High-dimensional continuous control using generalized advantage estimation