- An overview of gradient descent optimization algorithms

## Gradient Descent Variants

### SGD
随机梯度下降 Stochastic Gradient Descent，一次迭代使用单个样本或小批量(mini-batch)样本

$$
\begin{aligned}
    g_t =& \nabla_{\theta_{t-1}} J\left(\theta_{t-1}; x^{(i:i+n)}; y^{(i:i+n)}\right) \\
    \theta_t  =& \theta_{t-1} -\eta\cdot g_t    
\end{aligned}
$$

> $1 \le n \lt \text{batch_size}$

### BGD
批量梯度下降 Batch Gradient Descent, 一次迭代使用批量中所有样本

$$
\begin{aligned}
    g_t =& \nabla_{\theta_{t-1}} J\left(\theta_{t-1}; x; y\right) = \nabla_{\theta_{t-1}} \\
    \theta_t  =& \theta_{t-1} -\eta\cdot g_t    
\end{aligned}
$$

## Gradient Descent Optimizations


### Momentum
### NAG
Nesterov Accelerated Gradient

### AdaGrad
### Adadelta
### RMSProp
### Adam
#### Nadam
#### AdamW