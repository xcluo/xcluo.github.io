- doc2query-T5
- inverted indexes
- faiss-based implementation,



### MIPS算法
查询向量$q\in \mathbb{R}^{d}$，文档向量集合$\mathcal{X} = \{x_1, x_2, \dots, x_n\}$，其中$x_i \in \mathbb{R}^d$，目标是找到$\text{Top-}k=\argmax_{x \in \mathcal{X}} q^T x$

- 暴力计算复杂度为$O(nd)$，当$n$很大时，代价极高

#### Tree-based
#### Graph-based
HNSW（Hierarchical Navigable Small World）、NSG（Navigating Spreading-out Graph）
#### LSH
Locality-Sensitive Hashing局部敏感哈希
#### Quantization
PQ（Product Quantization）、IVFPQ（Inverted File with Product Quantization）。
#### Learned Indexes