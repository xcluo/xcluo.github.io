## XLM-RoBERTa
> 论文：Unsupervised Cross-lingual Representation Learning at Scale  
> Github: [fairseq](https://github.com/facebookresearch/fairseq)  
> FAAI, 2019 Nov, ACL 2020  


### 主要内容
- different languages using the same sampling distribution as Lample and Conneau (2019), but
with α = 0.3  
- 多语言上下文训练：同一批次（batch）中包含不同语言的样本，通过梯度更新隐式对齐语言间的语义空间。
- 共享词表与参数  
- 大规模数据覆盖：低资源语言通过高资源语言的参数共享获得迁移能力。
- TLM: translation language model
- wiki-100: using for XLM-100 and mBERT
- cc-100: using for XLM-R