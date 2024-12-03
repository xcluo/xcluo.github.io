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

3. 输出强分类器 $F(x) = f_T(x) = \sum_{i=0}^T \eta_t * h_t(x)$，因此

直接对损失函数求导，如$\mathcal{L} = -\ln \frac{1}{1+\exp^{-x}}$
   - $g=\frac{-\exp^{-x}}{1+ \exp^{-x}} = 1-\frac{1}{1+\exp^{-x}} = 1-\sigma(x)$
   - $h=-\frac{1}{1+\exp^{-x}} + -\frac{1}{(1+\exp^{-x})^2} = -\sigma(x) + \sigma(x)^2 = -\sigma(x)(1-\sigma(x))$
```python
In XGBoost, the first-order and second-order gradients are used for optimization during the training process. While decision trees themselves don't have straightforward gradients, XGBoost approximates these gradients by using the Taylor expansion of the loss function.

For the first-order gradient (the gradient of the loss function), it is often approximated using the negative gradient of the loss function with respect to the current prediction.

For the second-order gradient, XGBoost approximates the Hessian matrix, which represents the second-order partial derivatives of the loss function with respect to the predictions. This approximation is used to update the leaf values in the tree.

Here's a simple example to illustrate how these gradients are used in XGBoost:

Given a dataset with input features ( x ) and target ( y ), XGBoost starts with an initial prediction ( \hat{y}_0 ) for each sample.

For a given loss function (e.g., squared error loss), XGBoost computes the first-order gradient (negative gradient) and the second-order gradient for each sample based on the current predictions.

It then fits a decision tree to the negative gradient values, and for the second-order gradient, it approximates the Hessian matrix to update the leaf values in the tree.

The new tree captures the residual between the true target and the current prediction, and the leaf values are updated based on the Hessian approximation.

The updated tree is added to the ensemble, and the predictions are improved by considering the ensemble's combined output.

In summary, XGBoost uses the first-order and second-order gradients to guide the process of fitting decision trees, effectively optimizing the model's parameters to minimize the loss function.
```