### 集成学习


### Boosting
#### [LightGBM](Ensemble/Boosting/lightgbm.md)

### MoE
- 用存储性能换效果（时间效率影响持平）


### Dropout
[Dropout](Dropout/dropout.md)

### DAE
- mask部分input_token进行分类训练
- 部分input_token加入扰动噪声 -> 将离散的token通过加入噪声实现向量连续化，加强对抗性
- 一个d维token划分为k个d/k维token