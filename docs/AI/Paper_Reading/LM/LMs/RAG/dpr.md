## DPR
> 论文：**D**ense **P**assage **R**etrieval for Open-Domain Question Answering  
> FAIR & University of Washington & Princeton University 2020 Apr, EMNLP 2020  


#### 工作内容
1. 检索阶段
- query和document使用两个独立的BERT模型
- 对比学习负样本选取方案：1）Random，随机选取；2）BM25，基于BM25，高相关性但不含答案；3）Gold，其余问题的正样本
- inbatch negatives：in-batch中其余正样本作为负样本
- 本工作中使用了in-batch negatives + （1 BM25 negative + Gold）
- 使用1个BM25效果显著，使用2个及以上效果不明显
- 分别实现了BM25，DPR以及 BM25 + λ·sim(q, d)方案，第三种方案最优
- gpu dense用FAISS，cpu sparse用Lucene

2. 生成阶段
- 检索匹配，而不是检索生成
- 对比学习：1 positive passage + (m-1) negative passages from k candidate passages （基于BM25或PDR分数）
- 生成结果minor normalization：
    - Reading Wikipedia to answer opendomain questions
    - Latent retrieval for weakly supervised open domain question answering