### Features
#### TF-IDF

词频逆文档频率Term Frequency-Inverse Document Frequency，用于评估一个词在文档或语料库中的重要程度，这个数值的重要性随着它在一个文档中出现的次数成正比增加，但同时会随着它在整个文档集合中的普遍性成反比减少。

$$
\begin{aligned}
    &\text{TF}_{i, j} = \frac{\#word_{j}}{\#word\_in\_doc_i} \\
    &\text{IDF}_j =\log\bigg(\frac{\#doc + 1}{\#doc\_has\_word_j + 1}\bigg) \\
    &\text{TF-IDF}_{i, j} = \text{TF}_{i, j} \times \text{IDF}_{j} 
\end{aligned} 
$$

!!! info ""
    - $TF_{i, j}$ 为 $word_j$ 在 $doc_i$ 中出现的频率  
    - $IDF_{j}$ 为 $word_j$ 在所有文档中出现频率的倒数取对数值(1用于防止0值现象)，用于降低在多个文档中出现的词（如stop words）的权重，提升非常用词的权重。
    - <span style="color:red">$\text{TF-IDF}_{j}$ ?</span>

#### [MI](\AI\Paper_Reading\Trick\Ensemble\Ensemble\Boosting\lightgbm/#prechecker_features)

互信息Mutual Information，是度量两个随机变量之间相互依赖性(信息共享程度)的统计量。

$$
\text{mi} =\sum_{x \in X}\sum_{y \in Y} p(x,y)\log\frac{p(x,y)}{p(x)p(y)}
$$

- <span style="color:red">拆分为 $\text{mi}_{a}, \text{mi}_{b}, \text{mi}_{c}, \text{mi}_{d}$ 4部分</span>

#### [Chi-square statistic](\AI\Paper_Reading\Trick\Ensemble\Ensemble\Boosting\lightgbm/#prechecker_features)

卡方统计量 $\chi^2$ Chi-square statistic，用于衡量一个特征与类别标签之间的相关性强度，值越大表明相关性越高，0则表示无相关性。

$$
\chi^2 = \frac{n\times(ad - bc)^2}{(a+b)(c+d)(a+c)(b+d)}
$$

!!! info ""
    - **N11: a**, 表示同时具有两种属性的个体数量
    - **N10: b**, 表示具有第一个属性但不具有第二个属性的个体数量
    - **N01: c**, 表示不具有第一个属性但具有第二个属性的个体数量
    - **N00: d**, 表示同时不具有这两种属性的个体数量