https://www.jianshu.com/p/0355bafb26ae

macro：分别计算取平均
micro：加权

### Machine Learning
#### tf-idf

$$
\begin{aligned}
    &\text{tf-idf}_{i, j} = \text{tf}_{i, j} \times \text{idf}_{i} \\
    &\text{tf}_{i, j} = \frac{\#word_{i, j}}{\sum_{k}{\#word_{k, j}}} \\
    &\text{idf}_i =\log(\frac{\#doc}{1+\#doc\_has\_word_i}) \\
\end{aligned}
$$

#### [mi](\AI\Paper_Reading\Trick\Ensemble\Ensemble\Boosting\lightgbm/#prechecker_features) (mutual information)
$$
\text{mi} =\sum_{x \in X}\sum_{y \in Y} p(x,y)\log\frac{p(x,y)}{p(x)p(y)}
$$
#### [chi-square](\AI\Paper_Reading\Trick\Ensemble\Ensemble\Boosting\lightgbm/#prechecker_features)

$$
\text{chi} = \frac{n\times(ad - bc)^2}{(a+b)(c+d)(a+c)(b+d)}
$$

!!! info ""
    - **N11: a**, 表示同时具有两种属性的个体数量
    - **N10: b**, 表示具有第一个属性但不具有第二个属性的个体数量
    - **N01: c**, 表示不具有第一个属性但具有第二个属性的个体数量
    - **N00: d**, 表示同时不具有这两种属性的个体数量

### 分类
- macro-ROC：分别计算每类的ROC曲线再平均
- micro-ROC：对于每个样本，目标类别对应1，其余类别对应0，概率标签对位$(p_{y_i}, 1)$


### 生成