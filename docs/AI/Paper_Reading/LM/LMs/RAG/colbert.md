
## ColBERT
- 论文：ColBERT: Efficient and Effective Passage Search via **Co**ntextualized **l**ate Interaction over BERT  
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


- 长文档被压缩为单个向量，丢失细粒度语义
- 计算成本高，需预计算所有文档向量，难以动态更新索引

- 将查询和文档分别编码为多向量（每个token一个向量），在检索阶段计算细粒度匹配分数
- BucketIterator技术
#### Pruning
- 从 $k \ll N$ 文档中选取 top-k，对late interaction 进行剪枝操作
#### reranker
1. results reproduced by BM25
2. reranker: 执行MaxSim（sum of maximum similarity computations）

#### end-to-end
1. approximate stage: $N_q$ 个query token vector 访存N 个 document vector，分别返回 top-k$^{'}$ 个文档，结果$K = \text{unique}(N_q \times k^{'})$
2. reranker: 执行MaxSim（sum of maximum similarity computations）



## COIL
- project each Transformer—based token vector of query and document from  into low dimension