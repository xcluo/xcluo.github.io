## MRL
> 论文：**M**atryoshka **R**epresentation **L**earning  
> Github：[MRL](https://github.com/RAIVNLab/MRL)  
> University of Washington & Google Research & Harvard University, 2022 May, NeurIPS 2022

### 主要内容
- MRL: 该种方法是在bert后面接9个mlp层。mlp(768,8),(768,16),...(768,2048)。然后把bert编码得到的768维向量，在同时通过这9个mlp得到不同维度的向量，然后计算9个loss，累加起来进行训练。
- Efficient Matryoshka Representation Learning（MRL-E）：分别将前 8、16、...、2048维向量计算9个loss，累加起来进行训练