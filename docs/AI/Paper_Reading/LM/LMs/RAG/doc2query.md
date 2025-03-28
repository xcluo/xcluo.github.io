## Doc2query
> 论文：Document Expansion by Query Prediction  
> Github：[dl4ir-doc2query](https://github.com/nyu-dl/dl4ir-doc2query)  
> New York University & University of Waterloo & FAIR & Canadian Institute for Advanced Research 2019 Apr, CoRR 2019

### 主要内容
- 给定一个文档，让生成模型预测最可能会问哪些问题来提升QA搜索引擎的效果
- 第一个使用神经网络进行document expansion的工作
- Moses tokenizer标准化文本分词规则，文本清洗
- 当模型训练好使用top-k采样方法生成query
- 使用BM25进行检索，BERT作为reranker
- MS MARCO：reranker dataset
- 除去停用词，new_word：copied_word≈3：7
- without expansion MRR@10：18.4
- expand with new words MRR@10：18.8
- expand with copied words MRR@10：19.7
- expand with both words MRR@10：21.5
- RM3 query expansion反而会损伤效果