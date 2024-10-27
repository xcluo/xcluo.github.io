### RoBERTa
> 论文：RoBERTa: a **Ro**bustly optimized **BERT** pretraining **a**pproach  
> University of Washington & Facebook AI, NAACL-HLT 2019

- 动态mask，每次输入token seq时随机mask
- 取消nsp，只有MLM
- 扩大batch_size