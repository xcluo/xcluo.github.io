https://www.jianshu.com/p/0355bafb26ae

macro：分别计算取平均
micro：加权
### Decision Tree

#### ID3
1. Entropy

    $$
    \begin{aligned}
        E(X) =& -\sum_{k=1}^K p(k)\log p(k) \\
        =& -\sum_{k=1}^K \frac{\vert X^k \vert}{\vert X \vert}\log \frac{\vert X^k \vert}{\vert X\vert}
    \end{aligned}
    $$

    > $K$ 表示分类任务类别数  
    > $X$ 表示所有样本集合，$X^k$ 表示所有样本中属于 k-th 类别的样本集合

2. Information Gain  
    信息增益指选择某个特征进行分割能够减少的熵的程度

    $$
    \begin{aligned}
            IG(X, m) =&  E(X) - H(X\vert X_{m, v}) \\
            =& E(X) - \sum_{v \in V}\frac{\vert X_{m, v} \vert}{\vert X\vert}E(X_{m, v}) \\
            =& E(X) - \sum_{v \in V}\frac{\vert X_{m, v} \vert}{\vert X\vert}*\bigg(-\sum_{k=1}^k \frac{\vert X_{m, v}^k \vert}{\vert X_{m, v}\vert}\log \frac{\vert X_{m, v}^k \vert}{\vert X_{m, v}\vert}\bigg) \\
            = & E(X) -\sum_{v \in V}\sum_{k=1}^K\frac{\vert X_{m, v}^k \vert}{\vert X \vert}\log \frac{\vert X_{m, v}^k \vert}{\vert X_{m, v}\vert} \\ 
            = & E(X) -\sum_{v \in V}\sum_{k=1}^Kp(k, m=v)\log p(k\vert m=v)
    \end{aligned}
    $$

    > $m$ 表示 m-th 特征，$V=Set(X_m)$ 表示所有样本集合中 m-th 特征所有取值结合  
    > $X_{m, v}$ 表示所有样本中 m-th 特征等于v的样本集合  
    > $X_{m, v}^k$ 表示所有样本中 m-th 特征等于v且属于 k-th 类别的样本集合

#### C4.5
3. Information Gain Ratio

#### CART
1. Gini Index
    基尼指数
2. Gini Impurity

### Features
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

### Classification

| 预测值\真实值 | 正标签 | 负标签 |
| :-----:| :----: | :----: |
| 正标签 | TP | FP |
| 负标签 | FN | TN |

#### P、R、F1
=== "Precision"
    精确率，预测为正例样本中真实正例的百分比

    $$
    P=\frac{\text{TP}}{\text{TP} + \text{FP}}
    $$

=== "Recall"
    召回率，真实正例被预测为正例的百分比

    $$
    R=\frac{\text{TP}}{\text{TP} + \text{FN}}
    $$

=== "Accuracy"
    准确率，样本总数中被正确预测（包括负例）的百分比

    $$
    Accuracy=\frac{TP+TN}{TP+TN+FP+FN}
    $$

=== "F1-score"
    $$
    \begin{aligned}
    P &= \frac{\text{TP}_1 + \text{TP}_2 + ... + \text{TP}_k}{\text{TP}_1 + \text{TP}_2 + ... + \text{TP}_k + \text{FP}_1 + \text{FP}_2 + ... + \text{FP}_k} \\
    R &= \frac{\text{TP}_1 + \text{TP}_2 + ... + \text{TP}_k}{\text{TP}_1 + \text{TP}_2 + ... + \text{TP}_k + \text{FN}_1 + \text{FN}_2 + ... + \text{FN}_k} \\
    & \text{micro-F1} = \frac{2*P*R}{P+R} \\
    & \text{macro-F1} = \frac{\sum_1^k F1_k}{k}
    \end{aligned}
    $$


    !!! info ""
      - 多(大于2)分类中，对于micro-average，精确率P、召回率R、准确率Accuracy和F1是相等的  
      - micro更关注整体效果，适用于类别相对平衡的情况；  
      - macro更关注每个类别的效果，适用于类别不平衡的情况或需要评估每个类别的效果的情景；


=== "Fn-score"
    $$
    Fn=\frac{(n^2+1)*P*R}{n^2*P+R}
    $$

#### ROC、AUC
=== "ROC"
    Receiver Operating Characteristic Curve，受试者工作特征曲线，表示收益和成本之间的权衡关系：
    
      - 成本横坐标为$FPR=FP/N=FP/(FP+TN)$，即真实负例中被预测为正例的比例；
      - 收益纵坐标为$TPR=TP/P=TP/(TP+FN)$，即真实正例中被预测为正例的比例；
    
    曲线绘制（离散的plot图）与面积计算：

      1. 将P+N个样本的按目标类预测置信值降序排列，初始点位于(0, 0)；
      2. 遍历样本序列，以该点置信值为阈值t，计算tpr和fpr并描点连线（实际只描绘转向的点）；
      3. 绘制曲线即为ROC，曲线与横坐标轴围成的（分段矩形）区域面积为AUC；

    !!! info ""
        - 等价于从(0, 0)开始，顺序遍历遇见正例⬆️移动$1/P$，遇见负例➡️移动$1/N$，直至移动至(1, 1)。
        - micro-ROC：对于$n$个样本，$k$个类，共$n*k$个置信值降序排序，每个阈值计算tpr, fpr
        - macro-ROC绘制
            1. 得到各类的$(fpr_i, tpr_i)$点集并对横坐标去重得到$fpr$；
            2. 遍历各类的$(fpr_i, tpr_i)$获得$fpr$点集对应的$tpr$均值（各类[a, b)范围内的$tpr_i$为$frp_a$对应的$tpr_i$）
            3. 获得$(frp, tpr)$后即可进行ROC曲线绘制
        - micro-ROC更注重整体分类效果；
        - macro-ROC在处理不平衡数据集时，更能反应少数类别的分类性能（micro在排序过程中稀释了少数样本类）；
    

=== "AUC"
    Area Under Curve，曲线下面积，即ROC曲线与坐标轴围成的面积，取值范围处于[0, 1]，值越大表示表示效果越好。

    ```python
    from sklearn.metrics import roc_curve, auc
    # label_i: ROC + AUC
    fpr[i], tpr[i], thresholds = roc_curve(y[:, i], y_score[:, i])  # thresholds为降序排列的
                                                                    # 转折点（与上一状态相比变向）阈值
    roc_auc[i] = auc(fpr[i], tpr[i])

    # micro-ROC + AUC
    fpr["micro"], tpr["micro"], _ = roc_curve(y.flatten(), y_score.flatten())
    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

    # macro-ROC + AUC
    fpr["macro"] = np.unique(np.concatenate([fpr[i] for i in range(n_classes)])) # np.unique升序返回结果
    ''' 
        获取 `fpr["macro"]` 对应的 `tpr["macro"]=mean_tpr` 
        - 各类[a, b)范围内的`tpr_i`为左边界`frp_a`对应的`tpr_i``
    '''
    roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])
    ```

### Generation
$S$表示句子序列，$W$表示词或token

#### PPL
Perplexity，指模型在生成一段内容时的困惑程度，值越高，模型困惑度越大，反之越小越自信。

$$
\begin{aligned}
PPL &= \frac{1}{p_{\theta}(w_1w_2...w_n)^n} = \sqrt[-n]{p_{\theta}(w_1w_2...w_n)} \\
&=\sqrt[-n]{\prod_{t=1}^np_{\theta}(w_t|w_{\lt t})} \\
\log{PPL} &= -\frac{\sum_{i=t}^n\log{p_{\theta}(w_t|w_{\lt t})}}{n} \\
PPL &= \exp\Bigg(-\frac{\sum_{i=t}^n\log{p_{\theta}(w_t|w_{\lt t})}}{n}\Bigg) 
\end{aligned}
$$

!!! info ""
    在生成模型中，`ppl = nll.exp()`，即【负对数似然和的均值】作为指数



### Question Answering
#### Extract Match
$$
EM=\begin{cases}
1, & \text{if}\ S_{pred}\ =\ S_{ref} \\
0, & \text{if}\ S_{pred}\ \neq \ S_{ref} \\
\end{cases}
$$

#### F1

$$
\begin{aligned}
W_{match} &= W_{pred}\cap W_{ref} \\
P &= \frac{|W_{match}|}{|W_{pred}|} \\
R &= \frac{|W_{match}|}{|W_{ref}|} \\
F1 &= \frac{2*P*R}{P+R}
\end{aligned}
$$

!!! info ""
    $W_{match}$取的是`#!python Counter(pred_tokens) & Counter(ref_tokens)` 的就低token交集

### Translation & Summary
#### BLEU
Bilingual Evaluation Understudy，是用于评估自然语言的字句用机器翻译出来的品质（精度）的一种指标。

=== "BLEU"
    $$
    BLEU = \frac{\sum_{token \in C, S} Counter(token)}{len(C)}
    $$

    > 无次数就低约束，`C="the the the", S="the dog"`结果为`3/3=1`
    
=== "+ clip"
    $$
    BLEU = \frac{\sum_{token \in C, S} Counter_{match}(token)}{len(C)}
    $$

    > 增加就低约束，`C="the the the", S="the dog"`结果为`1/3=0.33`

=== "+ N-gram"
    $$
    \begin{aligned}
    BLEU_N &= \frac{\sum_{gram_N \in C, S} Counter_{match}(gram_N)}{\sum_{gram_N \in C}Counter(gram_N)} \\
    BLEU &= \sum_{n=1}^N w_n\log BLEU_n
    \end{aligned}
    $$

=== "+ BP"
    brevity penalty，增加简洁性约束，惩罚训练结果倾向短句的现象（缩小短句的BLEU值）。

    $$
    \begin{aligned}
    BP&=\begin{cases}
    1, & \text{if}\ c \gt r \\
    e^{1-r/c}, & \text{if}\ c \le r \\
    \end{cases} \\
    BLEU &= BP*\exp\Bigg(\sum_{n=1}^N w_n\log BLEU_n\Bigg) \\
    \log BLEU &= 1-\frac{r}{c} + \sum_{n=1}^N w_n\log BLEU_n
    \end{aligned}
    $$
    
    
!!! info ""
    - $w_n$为权重，一般为均匀加权，即$w_n=\frac{1}{N}$，$N$的上限取值为4。
    - 多个句子的BLEU计算时简单地通过累加操作增加相应的分子分母


#### ROUGE
Recall-Oriented Understudy for Gisting Evaluation，是评估摘要总结以及机器翻译效果（召回）的一组指标

=== "ROUGE-N"
    $$
    \text{RG-N} = \frac{\sum_{S\in Refer} \sum_{gram_N\in S} Counter_{match}(gram_N)}{\sum_{S\in Refer} \sum_{gram_N\in S} Counter(gram_N)}
    $$

=== "ROUGE-L"
    $$
    \begin{aligned}
    P_{lcs} &= \frac{LCS(C, S)}{len(C)} \\
    R_{lcs} &= \frac{LCS(C, S)}{len(S)}\\
    \text{RG-L} &= F_{lcs} = \frac{(1+\beta^2)P_{lcs}R_{lcs}}{\beta^2 P_{lcs} + R_{lcs}}
    \end{aligned}
    $$

=== "ROUGE-W"
    [sss](https://zhuanlan.zhihu.com/p/659637538)
    https://blog.csdn.net/BIT_666/article/details/132347794

=== "ROUGE-S"
    sss

!!! info ""
    - ROUGE取值范围为[0, 1]；
    - $Refer$为参考文本序列集合；
    - N表示N-gram，一般取值为1，2，3，$Counter_{match}$为就低操作；
    - $C$表示生成文本序列，$S$为参考文本序列，$len()$返回序列token数；
    - L表示最长公共子序列Longest common subsequence（==注意不是最长连续公共子序列==）；


### Search
#### MRR
Mean Reciprocal Rank，对搜索算法进行评估的机制，MRR是指多个查询语句的排名倒数的均值

$$
MRR = \frac{1}{|Q|}\sum_{i=1}^{|Q|}\frac{1}{rank_i}
$$


- Pearson/Spearman correlation Coefficient，有道云笔记
- Matthews correlation Coefficient，有道云笔记
- Lp距离，有道云笔记
