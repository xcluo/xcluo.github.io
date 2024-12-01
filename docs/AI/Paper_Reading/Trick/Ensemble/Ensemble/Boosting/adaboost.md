### AdaBoost
> 论文：A Decision-Theoretic Generalization of On-Line Learning and an Application to Boosting  
> AdaBoost: **Ada**ptive **Boost**ing  
> AT&T Labs, EuroCOLT 1997

#### 基本原理
1. **初始化**，训练集$D=\{(x^i, y^i)\}_{i=1}^n$，最大迭代轮数$T$，各样本权重$w_0^i=\frac{1}{n}$

    > $y_i\in \{-1, +1\}$ 分别对应负、正类

2. **迭代训练**，对于当前迭代轮次$t$：
    - 基于样本权重 $W_{t}$ 加权loss来训练弱学习器$f_t(x)$
    - 结合 $W_{t}$ 计算 $f_t(x)$ 在训练集上的分类错误率(error rate) $e_t$

        $$
        e_t = \sum_{i=1}^n w_t^i*\mathbb{I}\big(f_t(x^i)\ne y^i\big)
        $$

        > $\mathbb{I}$ 为指示函数，如果条件成立则1，反之则0。

    - 计算弱学习器 $f_t(x)$ 的权重 $\alpha_t$，错误率越小弱学习器权重越大
        
        $$\alpha_t=\frac{1}{2}\ln\bigg(\frac{1-e_t}{e_t}\bigg)$$

    - 更新各样本权重并归一化，预测正确降低权值，预测错误提升权值

        $$
        w_{t+1}^i=\frac{w_t^i * \exp^{-\alpha_t*y_i*f_t(x_i)}}{\sum_k^n w_t^k * \exp^{-\alpha_t*y_k*f_t(x_k)}}
        $$    

3. **组合弱学习器**，$F(x)=\text{sign}\Big(\sum_{t=1}^T\alpha_t*f_t(x)\Big)$

    > $\text{sign}$ 用于取标签，也可以去除这部分获取logit值进而调整阈值