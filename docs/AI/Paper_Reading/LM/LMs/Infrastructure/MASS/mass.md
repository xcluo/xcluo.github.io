### MASS
> 论文：MASS: **MA**sked **S**equence to **S**equence pre-training for language generation  
> Nanjing University of Science and Technology & MSR, ICML 2019


#### 工作亮点
- encoder + decoder
- encoder input: $x^{\backslash u:v}$
- decoder predict: $x^{u:v}$
- hpyerparameter $k=v-u+1$，其中$k=0.5*m$效果相对最好，太大或大小可能会更偏向NLG和NLU
- 连续mask效果比离散mask效果更好，mass在decoder只输入待预测的部分比全部输入效果更好