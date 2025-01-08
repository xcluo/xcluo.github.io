#### 余弦相似度Cosine Similarity
1. cosiine similarity

    $$
    \text{cos_sim}(x, y) = \frac{x\cdot y}{\vert x \vert * \vert y \vert}
    $$

2. adjusted cosine similarity，进一步考虑了向量均值的差异

    $$
    \text{adj_cos_sim}(x, y) = \frac{(x-\bar{x})\cdot(y-\bar{y})}{\vert x-\bar{x}\vert * \vert y-\bar{y} \vert}
    $$

    > $\bar{\cdot}$ 表示向量均值

#### KL散度Kullback-Leibler Divergence
#### 皮尔斯相关系数Pearson Correlation Coefficient
#### 斯皮尔曼秩相关系数Spearman's Rank Correlation Coefficient
#### Lp距离Lp Distance