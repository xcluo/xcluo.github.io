### XGBoost
> 论文：XGBoost: A Scalable Tree Boosting System  
> XGBoost:e**X**treme **G**radient **Boost**ing  
> Tianqi Chen & University of Washington, SIGKDD 2016

#### 基本原理
本质上还是GBDT，对于当前迭代轮次$t$，$f_t(x)=f_{t-1} + h_t(x)$：

- 目标loss函数包括经验损失和正则项

    $$
    \begin{aligned}
        \mathcal{L}&=\sum_{i=1}^n L\big[f_t(x^i), y^i\big] + \sum_{j=1}^{t}\Omega(h_j) \\
        &=\sum_{i=1}^n L\big[f_{t-1}(x^i) + h_t(x^i), y^i\big] + \sum_{j=1}^{t}\Omega(h_j) \\
        基于& \text{Taylor}  公式，将f_{t-1}(x^i)视作x_0，h_t(x^i) 视作 \Delta x，得 \\
        &\approx \sum_{i=1}^n \big(L[f_{t-1}(x^i), y^i] + \frac{\partial L[f_{t-1}(x^i), y^i]}{\partial f_{t-1}(x^i)}h_t(x^i) + \frac{\partial^2 L[f_{t-1}(x^i), y^i]}{\partial^2 f_{t-1}(x^i)}h_t^2(x^i) \big) + \sum_{j=1}^{t}\Omega(h_j) \\ 
        由于&只关注轮次t的弱分类器h_t(x)，因此最终损失可简化为 \\
        \mathcal{L}&=\sum_{i=1}^n \bigg(\frac{\partial L[f_{t-1}(x^i), y^i]}{\partial f_{t-1}(x^i)}h_t(x^i) + \frac{\partial^2 L[f_{t-1}(x^i), y^i]}{\partial^2 f_{t-1}(x^i)}h_t^2(x^i) \bigg) + \Omega(h_t) + c_t \\
        &= \sum_{i=1}^n \big(g_ih_t(x^i) +  \hbar_ih_t^2(x^i) \big) + \Omega(h_t) + c_t 
    \end{aligned}
    $$

    > $g_i=\frac{\partial L[f_{t-1}(x^i), y^i]}{\partial f_{t-1}(x^i)}$ 为$t-1$轮次目标残差对分类器的一阶导数  
    > $\hbar_i=\frac{\partial^2 L[f_{t-1}(x^i), y^i]}{\partial^2 f_{t-1}(x^i)}$ 为$t-1$轮次目标残差对分类器的二阶导数    
    > $c_t$ 为前$t-1$轮所有弱分类器的残差与正则项的总和

- 其中正则项计算为

    $$
    \begin{aligned}
    \Omega(h) =& \gamma M + \frac{1}{2}\lambda \Vert w\Vert^2_2 \\
    =& \gamma M + \frac{1}{2}\lambda\sum_{j=1}^{M} w_j^2
    \end{aligned}
    $$
    > $M$ 为决策树 $h$ 的叶节点数，$w \in \mathbb{R}^M$ 为决策树各叶节输出的回归值构成的向量  
    > **每个节点对应于一个类别(或类别集合)，对于输入$x^i$，各节点会输出回归值$w^i_{j}$**  
    >  $\gamma$ 和 $\lambda$ 为超参数  

- 进一步简化目标loss函数

    $$
    \begin{aligned}
        \mathcal{L}  &= \sum_{i=1}^n \big(g_ih_t(x^i) +  \hbar_ih_t^2(x^i) \big) + \gamma M + \frac{1}{2}\lambda\sum_{j=1}^{M} w_j^2 + c_t \\
        &= \sum_{j=1}^M\Big[ (\sum_{i \in I_j} g_i)w_j + \frac{1}{2}(\sum_{i \in I_j} \hbar_i + \lambda)w_j^2 \Big] + \gamma M + c_t  \\
        根据&二元一次方程特性（\text{loss} \ge 0，开口朝上），可得最小损失为 \\
        \mathcal{L}_t^{opt} &= -\frac{1}{2}\sum_{j=1}^M\frac{(\sum_{i\in I_j} g_i)^2}{\sum_{i\in I_j }h_i + \lambda} + \gamma M
    \end{aligned}
    $$

    > 其中$I_j=\{x^i\vert q(x^i)=j\}$ 表示最终分类在决策树节点$j$上的样本集  
    > $q$ 为决策树

    