#### Cosine Similarity
余弦相似度，即两个向量夹角的cos值，取值为[-1, 1]

1. Cosiine Similarity

    $$
    \text{cos_sim}(x, y) = \frac{x\cdot y}{\vert x \vert * \vert y \vert}
    $$

2. Adjusted Cosine Similarity，进一步考虑了向量均值的差异

    $$
    \text{adj_cos_sim}(x, y) = \frac{(x-\bar{x})\cdot(y-\bar{y})}{\vert x-\bar{x}\vert * \vert y-\bar{y} \vert}
    $$

    > $\bar{\cdot}$ 为向量均值

#### Pearson Correlation Coefficient
皮尔森相关系数PCC，也称为皮尔森积矩相关系数（Pearson Product-Moment Correlation Coefficient，PPMCC），用来衡量两个变量间线性关系强度，取值为[-1, 1]表示`完全负相关 → 不相关 → 完全正相关`，（假设数据服务正态分布）本质为两个向量中心化后的[cosine similarity](#cosine-similarity)：

$$
r(x, y) = \frac{\text{Cov}(x, y)}{\sigma_x \sigma_y} = \frac{\sum_{i=1}^n(x_i-\bar{x})(y_i-\bar{y})}{\sqrt{\sum_{i=1}^n(x_i - \bar{x})^2}\sqrt{\sum_{i=1}^n(y_i - \bar{y})^2}}
$$

#### Lp Distance
也称作闵可夫斯基距离Minkowski Distance，用于衡量两个向量间距离，计算方式为

$$
L_p(x, y) = \bigg(\sum_{i=1}^n \vert x_i - y_i\vert^p\bigg)^{\frac{1}{p}}
$$

- $L_0$ Distance，也叫汉明距离Hamming Distance，表示各维度上元素值不同的数量

    $$
    L_0(x, y) = \sum_{i=1}^n \mathbb{I}(x_i \ne y_i)
    $$

- $L_1$ Distance，也叫曼哈顿距离Manhattan Distance

    $$
    L_1(x, y) = \sum_{i=1}^n \vert x_i - y_i\vert
    $$

- $L_2$ Distance，也叫欧几里得距离Euclidean Distance

    $$
    L_2(x, y) =\sqrt{\sum_{i=1}^n (x_i - y_i)^2}
    $$

- $L_\infty$ Distance，也叫切比雪夫距离Chebyshev Distance，表示所有维度上的最大绝对值差

    $$
    L_\infty(x, y) = \max (\vert x_1 - y_1 \vert, \vert x_2 - y_2 \vert, \dots, \vert x_n - y_n \vert)
    $$

!!! info ""
    计算范数 $L_n$ norm($L_\infty$ norm为无穷范数)时，$y$ 值设为全0

#### Kullback-Leibler Divergence
KL散度，也叫做相对熵Relative Entropy或信息散度Information Divergence，描述两个概率分布间差异的**非对称性**度量，$D_{KL}(P\Vert Q)\ge 0$，值越大表示差异性越大。

- 离散概率分布 $P$ 和 $Q$ 的KL散度：

    $$
    D_{KL}(P\Vert Q) = \sum_{i=1}^n P(x_i)\log \frac{P(x_i)}{Q(x_i)} = \sum_{i=1}^n \big(-P(x_i)\log {Q(x_i)} \text{+} P(x_i)\log P(x_ix)\big)
    $$

- 连续概率分布 $P$ 和 $Q$ 的KL散度：

    $$
    D_{KL}(P\Vert Q) = \int_{-\infty}^{\infty} P(x)\log \frac{P(x)}{Q(x)}
    $$


#### Spearman's Rank Correlation Coefficient
斯皮尔曼等级相关系数，通常简称为斯皮尔曼相关系数，用于衡量两个变量之间的单调关系而非精确的线性关系

$$
\rho(x , y) = 1 - \frac{6\sum_{i=1}^n d_i^2}{n(n^2-1)} = 1 - \frac{6\sum_{i=1}^n (r_{x_i} - r_{y_i})^2}{n(n^2-1)}
$$

> $d_i$ 表示向量 $x$ 和 $y$ 中 $i\text{-}th$ 元素argsort排序后的位次差值