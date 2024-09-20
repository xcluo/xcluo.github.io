### LoRA
> 论文：LoRA: Low-rank Adaptation of large language models  
> Github: [LoRA-GA: Low-Rank Adaptation with Gradient Approximation](https://github.com/microsoft/LoRA)  
> MicroSoft, ICLR, 2022

#### 工作要点
1. 冻结预训练模型的参数，在每层额外插入trainable rank decomposition matrices
2. on GPT-3 175B, trainable参数量减少为10000分之一，GPU使用减少为3分之一
3. 效果和直接训练差不多，且infer性能消耗区别不大