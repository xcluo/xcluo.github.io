- inverted indexes
- faiss: facebook ai similarity search
- Anserini IR toolkit

### query expansion
- splade
### document expansion
generate query from document
#### doc2query
#### docT5query
- to train a model, that when given an input document, generates questions that the document might answer
### 词级别交互
- colbert




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