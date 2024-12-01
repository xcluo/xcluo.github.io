### GBM
> 论文：Greedy Function Approximation: A Gradient Boosting Machine  
> GBM: **G**radient **B**oosting **M**achines  
> Stanford University, Annals of Statistics 2001


#### 基本原理
**每次迭代**中，新的模型被训练来**拟合当前模型预测值与真实值之间的残差**。训练集 $D=\{(x^i, y^i)\}_{i=1}^n$，最大迭代轮数 $T$

1. **初始模型**，使用一个简单模型作为初始模型 $f_0(x)$ ，通常是一个常数模型 
    
    $$
    f_0(x)=\text{arg }\mathop{\text{min}}\limits_{\gamma}\sum_{i=1}^n L(y_i, \gamma)
    $$

2. **迭代训练**，使用新的模型来修正上一轮的误差，对于当前迭代轮次$t$：
    - 计算当前模型 $f_{t-1}$ 与目标值 $y^i$ 残差的负梯度

        $$
        \tilde{y}^i = -\frac{\partial L\big(y^i - f_{t-1}(x^i)\big)}{\partial f_{t-1}(x^i)}
        $$

        

    - 训练模型 $h_t(x)$ 来**拟合残差负梯度**

    - 将残差梯度拟合模型整合入现有模型中 (可视作对上一轮次模型进行梯度下降)
        
        $$
        f_t(x) =f_{t-1}(x) + \eta_t * h_t(x)
        $$

        > $\eta_t$ 为学习率

    - 评估是否达到某个停止准则，如最大迭代轮次、性能不再提升、残差小于一定阈值。

3. 输出强分类器 $F(x) = f_T(x) = \sum_{i=0}^T \eta_t * h_t(x)$
