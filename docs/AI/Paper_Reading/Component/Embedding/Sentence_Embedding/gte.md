## GTE
> 论文：Towards **G**eneral **T**ext **E**mbeddings with Multi-stage Contrastive Learning  
> Alibaba Group, 2023 Aug

### 主要内容
GTE，阿里巴巴达摩院推出

- multi-stage contrastive learning pipeline  
    1. unsupervised text pairs from various data sources for contrastive pre-training
    2. high-quality supervised text pairs with human labels from multiple sources for contrastive fine-tuning
    3. Furthermore, since our model is trained using code data as well

- using a large batch size is crucial to better model performance by reducing the gap between training and inference
#### Unsupervised CL
- CPT, Contrastive Pre-Training
- unsupervised pre-training
- we exclusively utilized open-source data and did not employ any filtering or cleaning
methods. details in Appendix A
- text pair format including (title, body), (title, abstract), (citation, reference), (post, comment), (entity, description), (question, answer), (summary, content), (text, code)
- 788M pairs
![alt text](image-3.png)

- **data sources often differ significantly** in terms of the number of training instances. To address this imbalance, 对各个data source进行多项式采样$p_i = \frac{n_i^{\alpha}}{\sum_j n_j^{\alpha}}$
- 每个batch中的任务类型确保相同
- 大batch_size非常有必要
- 分布式并行训练，large batch_size, set max_seq_length=128
- pretrained models were initialized using the corresponding size MiniLM/BERT models
#### Supervised CL
- supervised fine-tuning
- two pieces of text and optional hard negatives mined by an extra retriever to form text triples.
- handle both symmetric tasks (e.g., semantic textual similarity) and asymmetric tasks (e.g., passage retrieval), collecting large variety of tasks and domains. details in Appendix A
- ∼3M pairs for fine-tuning
- a large batch size is unnecessary since hard negatives can already provide a reliable gradient estimation of the learning objective
- batch_size=128, 16 contrastive samples (1 positive + 1 hard + remaining in-batch random)
- max_seq_length=512

#### special optimization strategies
- enlarges the negative samples with both in-batched queries and documents
- amp with fp16, deepspeed zero, gradient checkpointing

#### Ablation

## mGTE
> 论文：mGTE: Generalized Long-Context Text Representation and Reranking Models for Multilingual Text Retrieval  
> Alibaba Group, 2024 Jul，EMNLP 2024


## Qwen3 Embedding
> Github：[Qwen3-Embedding](https://github.com/QwenLM/Qwen3-Embedding)

- https://qwenlm.github.io/blog/qwen3-embedding/