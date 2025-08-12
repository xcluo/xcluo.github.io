## LoRA-GA
> 论文：AdaLoRA: adaptive budget allocation for parameter-effecient fine-tuning
> AdaLoRA: **Ada**ptive **Lo**w **R**ank **A**dapation  
> Github：[AdaLoRA](https://github.com/QingruZhang/AdaLoRA)  
> MicroSoft, 2023 Mar, ICLR 2023

### 主要内容
- https://zhuanlan.zhihu.com/p/657130029

- train 1.5倍LoRA，infer等同lora
- 动态分配rank给每层的每个矩阵
- 越后层相对来说rank分配越多
- ablation消融实验表明正则项和SVD的效果和必要性，约束P和Q为正交矩阵且互不依赖
- 不同task的分布不同
- only LoRA q and v, LoRA all weight matrix improved, table 14
- 训练的性能开销，11%-16% incur