### GBDT
> 论文：Greedy Function Approximation: A Gradient Boosting Machine  
> GBDT: **G**radient **B**oosting **D**ecision **T**rees  
> Stanford University, Annals of Statistics 2001

#### 基本原理
也称为 Gradient Boosted Regression Trees (GBRT) 或者是 Gradient Tree Boosting，是GBM的一种特例，其中：  

- 每个弱学习器 $h_t(x)$ 都是CART决策树。