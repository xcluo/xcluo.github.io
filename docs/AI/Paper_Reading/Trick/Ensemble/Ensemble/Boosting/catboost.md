### CatBoost
> 论文: CatBoost: unbiased boosting with categorical features  
> CatBoost: **Cat**egorical **Boost**ing  
> Yandex & Moscow Institute of Physics and Technology, NIPS 2017  


### 基本原理
- https://mp.weixin.qq.com/s/iYumC_JlMHZpBuAd4ryWFw
- categrorical features processing，处理【离散】的类别型特征为新的数值型特征
- categrorical features 是一个包含互相独立的离散特征集，常见的有one-hot encoding，对每个特征新增一个二分特征进行表示，但对于某些具有id属性的特征，二分属性往往不够，而是需要大量的枚举值作为区分
    - 避免高维稀疏矩阵或信息丢失
- drift: category feature 在训练集和测试集上求得的值不一致（如某一特征取值所属类别的占比）
- TS (target statistics): greedy TS, Holdout TS[x], Leave-one-out TS, Ordered TS，基于每个类别的信息来计算目标特征值


- greedy TS
    - $\hat{x}_i^k=\frac{\sum_{x^j \in D_k} \mathbb{1}_{\{x^j_i = x^k_i\}}\cdot y^j + ap}{\sum_{x^j \in D_k} \mathbb{1}_{\{x^j_i = x^k_i\}} + a}$
        - $D_k \subset D \backslash \{x^k\}$, excluding $x^k$ to avoid target leakage

- row-wise ordered TS，利用之前的历史数据来估计该类别与目标特征值之间的相关性
  - $\hat{x}_i^k=\frac{\sum_{x^j \in D_k} \mathbb{1}_{\{x^j_i = x^k_i\}}\cdot y^j + ap}{\sum_{x^j \in D_k} \mathbb{1}_{\{x^j_i = x^k_i\}} + a}$  
      - train step: $D_k = \{x_j: j\lt k\}$, historical instances to avoid target leakage并减缓过拟合
      - infer step: $D_k = D$

- gradient bias
- prediction shift：训练时标签信息泄露了，使用了本该被预测的值；测试或验证时输入的是未参与训练的数据进行预测，两者存在差异。此外就是每次迭代时，使用的数据集(样本顺序和内容)是相同的
- Analysis of prediction shift
- Feature combinations
- plain boosting: ordered TS + gbdt
- ordered boosting (Algorithm 1 & Figure 1)
    1. 打破数据固有的顺序依赖性，random row/instance permutation
    2. 对每一个样本 都训练一个单独的模型 
    3. 返回$F_{n}^{T}$（可用动态规划dp理解或Causal mask）

- catboost (Algorithm 2): combine ordered TS and ordered boosting
    1. 分块处理，将随机排列后的数据分成多个不相交小块$B_1, B_2, \dots$，每个块基于之前历史块信息计算ordered TS，顺序更新
    - 从排列$\sigma_1, \dots, \sigma_n$ 随机选取一个用于决定决策树中内部节点的分裂
    - 使用 $\sigma_0$ 来确定决策树各叶子节点的值
    - 使用排列 $\sigma_1, \dots, \sigma_n$ 的平均梯度值以梯度下降更新模型

- oblivious decision trees，decision table
- https://blog.csdn.net/Water8L/article/details/138172853