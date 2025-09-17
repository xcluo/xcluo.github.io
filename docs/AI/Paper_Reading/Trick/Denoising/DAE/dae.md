### DAE
- mask部分input_token进行分类训练
- 部分input_token加入扰动噪声 -> 将离散的token embedding通过加入噪声实现向量连续化，加强对抗性
- 一个d维token划分为k个d/k维token，再引入局部性造成，常见于图像patch或者或者向量表示