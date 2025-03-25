
## ColBERT
- 论文：ColBERT: Efficient and Effective Passage Search via **Co**ntextualized **L**ate Interaction over BERT  
- Stanford University 2020 Apr, SIGIR 2020

### 主要内容
#### query encoder

#### document encoder


#### Late Interaction
query-doc的交互计算推迟到最后一层

$$
S_{q, d} = \sum_{i=1}^{\vert q \vert} \max_{j = 1}^{\vert d \vert} E_{q_{i}}\cdot E_{d_j}^T
$$

> 保留词级别匹配的细粒度语义  
> $E_d$ 可离线预计算，加速在线检索
> 对比学习<q, d+, d->
#### Pruning