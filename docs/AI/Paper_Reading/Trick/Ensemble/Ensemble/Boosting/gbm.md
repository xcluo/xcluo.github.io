### GBM
> 论文：Greedy Function Approximation: A Gradient Boosting Machine  
> GBM: **G**radient **B**oosting **M**achines  
> Stanford University, Annals of Statistics 01


#### 基本原理
训练集 $D=\{(x^i, y^i)\}_{i=1}^n$，最大迭代轮数 $T$

1. **初始模型**，使用一个简单模型作为初始模型 $f_0(x)$ ，通常是一个常数模型 
    
    $$
    f_0(x)=\text{arg }\mathop{\text{min}}\limits_{\gamma}\sum_{i=1}^n L(y_i, \gamma)
    $$

2. **迭代训练**，使用新的模型来修正上一轮的误差，对于当前迭代轮次$t$：
    - 计算当前模型 $f_{t-1}$ 的负梯度

3. 输出强分类器 $F(x) = f_T(x)$