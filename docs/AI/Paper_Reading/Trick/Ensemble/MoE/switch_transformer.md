- https://huggingface.co/blog/zh/moe#%E9%83%A8%E7%BD%B2%E6%8A%80%E6%9C%AF
### Switch Transformer
> 论文：Switch Transformers: Scaling to Trillion Parameter Models
with Simple and Efficient Sparsity  
> Google, JMLR 2022

#### 工作要点
1. selective precision training enables bf16 training
2. initialization schema for larger number of experts training
3. increased expert regularization improve
- overflow, referred to dropped tokens, skip and passed directly by residual connectation
- load blancing loss
    - 理想状态下$f_1$和$P_i$都应该是$\frac{1}{N}$，即tokens均匀分布于各experts，$\sum_{i=1}^Nf_i\cdot P_i=N*(\frac{1}{N}\cdot\frac{1}{N})=\frac{1}{N}$，因此需要额外乘以$N$（进行归一化来）消除专家数目不同带来的影响  
- 混合精度可以加速模型且效果相当：switch moe部分float32，其余部分bfloat16
    - 单纯用bfloat16模型会发散
- smaller parameter intialization for stability：$\mu=0,\sigma=\sqrt{s/n}$，$n$为input_dim, s为标量超参，此处取0.1
- smaller dropout on no-expert layers(0.1) and larger dropout on expert layers(0.4) performs better
- distilling llm into small dense model, mixture of hard and soft label: 0.25 of teacher and 0.75 of ground truth label，保留30%的提升效果  
- experts ff, experts attention
- scaling law: Scaling laws for neural language models  