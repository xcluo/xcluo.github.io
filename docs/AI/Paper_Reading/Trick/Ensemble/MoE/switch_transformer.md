- https://huggingface.co/blog/zh/moe#%E9%83%A8%E7%BD%B2%E6%8A%80%E6%9C%AF
### Switch Transformer
> 论文：Switch Transformers: Scaling to Trillion Parameter Models
with Simple and Efficient Sparsity  
> Google, JMLR 2022

#### 主要要点
1. selective precision training enables bf16 training
2. initialization schema for larger number of experts training
3. increased expert regularization improve
- overflow, referred to dropped tokens, skip and passed directly by residual connectation

- scaling law: Scaling laws for neural language models