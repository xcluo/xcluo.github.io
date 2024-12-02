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
        &\approx \sum_{i=1}^n \big(L[f_{t-1}(x^i), y^i] + \frac{\partial L[f_{t-1}(x^i), y^i]}{\partial f_{t-1}(x^i)}h_t(x^i) + \frac{1}{2}\frac{\partial^2 L[f_{t-1}(x^i), y^i]}{\partial^2 f_{t-1}(x^i)}h_t^2(x^i) \big) + \sum_{j=1}^{t}\Omega(h_j) \\ 
        由于&只关注轮次t的弱分类器h_t(x)，因此最终损失可简化为 \\
        \mathcal{L}&=\sum_{i=1}^n \bigg(\frac{\partial L[f_{t-1}(x^i), y^i]}{\partial f_{t-1}(x^i)}h_t(x^i) + \frac{1}{2}\frac{\partial^2 L[f_{t-1}(x^i), y^i]}{\partial^2 f_{t-1}(x^i)}h_t^2(x^i) \bigg) + \Omega(h_t) + c_t \\
        &= \sum_{i=1}^n \big(g_ih_t(x^i) +  \frac{1}{2}\hbar_ih_t^2(x^i) \big) + \Omega(h_t) + c_t 
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
        \mathcal{L}  &= \sum_{i=1}^n \big(g_ih_t(x^i) +  \frac{1}{2}\hbar_ih_t^2(x^i) \big) + \gamma M + \frac{1}{2}\lambda\sum_{j=1}^{M} w_j^2 + c_t \\
        &= \sum_{j=1}^M\Big[ (\sum_{i \in I_j} g_i)w_j + \frac{1}{2}(\sum_{i \in I_j} \hbar_i + \lambda)w_j^2 \Big] + \gamma M + c_t  \\
        根据&二元一次方程特性（\text{loss} \ge 0，开口朝上），可得最小损失为 \\
        \mathcal{L}_t^{opt} &= -\frac{1}{2}\sum_{j=1}^M\frac{(\sum_{i\in I_j} g_i)^2}{\sum_{i\in I_j }\hbar_i + \lambda} + \gamma M
    \end{aligned}
    $$

    > 其中$I_j=\{x^i\vert q(x^i)=j\}$ 表示最终分类在决策树节点$j$上的样本集  
    > $q$ 为决策树

- 决策时叶节点分裂的损失增益(损失值前后差异值)  

    $$
    \begin{aligned}
        \text{Gain} =& \mathcal{L}_{previous} - \mathcal{L}_{subsequent} \\
        =& \frac{1}{2}\bigg[\frac{(\sum_{i\in I_L} g_i)^2}{\sum_{i\in I_L }\hbar_i + \lambda} + \frac{(\sum_{i\in I_R} g_i)^2}{\sum_{i\in I_R }\hbar_i + \lambda} - \frac{(\sum_{i\in I} g_i)^2}{\sum_{i\in I }\hbar_i + \lambda}\bigg] - \gamma
    \end{aligned}
    $$

    ```python title="exact_greedy_split_finding"
    def exact_greedy_split_finding(I, d):
        """
        遍历所有特征找出最佳分裂点(分裂特征，特征分别值)
        args:
            I: instance set of current leaf node
            d: instance feature dimension
        return:
            split with max score by G_L and G_R
        """
        gain = 0
        G, H = sum([g_i for i in I]), sum([hbar_i for i in I])
        for k in range(1, d+1):                         # 遍历特征
            G_L = 0, H_L = 0
            for i in sorted(I, key=lambda x: x[k]):     # 按当前特征对样本集排序
                G_L, H_L = G_L + g_i, H_L + hbar_i
                G_R, H_R = G - G_L, H - H_L
                gain = max(gain, G_L**2/(H_L + λ) + G_R**2/(H_R + λ) - G**2/(H + λ))

        return gain
    ```
    > 枚举思路，效率不佳

    ```python title="approximate_split_finding"
    def approximate_split_finding(I, d):
         """
        enumerates over all the possible splits on all the features to find the best split
        args:
            I: instance set of current leaf node
            d: instance feature dimension
            N: the number of approximate buckets of splitting the instance set on corresponding feature weight quantile
        return:
            split with max score by G_L and G_R
        """
        for k in range(1, d+1):
            S[k] = [s_k[i] for i in range(N)]            # 获取样本集在当前特征N等分的分位点
        for k in range(1, d+1):
            G_kv = sum([g_j for j in I if ])
            H_kv = sum()

        以桶为基本元素进行G_L和G_R划分并计算max_score
    ```

    $$
        r_k(z) = \frac{\sum_{(x, k)\in D_k, x\lt z} \hbar}{\sum_{(x, k)\in D_k} \hbar} \\
        \vert r_k(s_{k, j}) - r_k(s_{k, j+1}) \vert \lt \epsilon, s_{k1} = \min_i x_{ik}, s_{kl} = \max_{i} x_{i, k}
        
    $$

    > $\epsilon$ 为分位置误差，因此每个bucket中约有 $1/\epsilon\approx N$ 个候选样本，基于 $\hbar$ 进行 N 等分

    因为loss都由 $\hbar$ 加权输出，所以使用 $1/\hbar$ 进行分位数划分基准  

    $$
    \begin{aligned}
        \mathcal{L} =& \sum_{i=1}^n \big(g_ih_t(x^i) + \frac{1}{2}\hbar_i h_t^2(x^i)\big) + \Omega(h_t) + c_t \\
        = & \frac{1}{2}\hbar_i\sum_{i=1}^n \big(h_t(x^i) + g_i/\hbar_i\big)^2 + \Omega(h_t) + c_t^{'}
    \end{aligned}
    $$

    实际应用中，对于样本x，具有稀疏特征是极其正常的，可以采用以下方法对缺值样本进行处理：
      1. 只对有值样本进行排序，随后将所有缺值样本分别放入sorted 最左侧和左右侧
      2. 遍历两种处理方案的结果，选择最佳(max_gain最大)的方案
    
    ```python title="sparsity-aware_split_finding"
    
    ```

    - 列特征子采样，一般取0.8
    - Shrinkage通过减小每棵树的贡献（即乘以学习率$\eta$，缺省为0.3），使模型更平滑，有助于防止过拟合。