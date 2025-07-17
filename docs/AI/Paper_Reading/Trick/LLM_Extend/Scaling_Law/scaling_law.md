- forward：正常矩阵运算flops包括乘法和加法，如 $W\in \mathbb{R}^{m\times n}, X\in \mathbb{R}^{bs\times m}$，因此运算$AB$的浮点数运算数为$2*bs*m*n$
- backwrd：反向传播过程一般为正向传播过程的2倍，分别对权重参数 $W \in \mathbb{R}^{m\times n}$ 和输入 $X \in \mathbb{R}^{bs \times m}$ 进行梯度计算，即 $\frac{\partial L}{\partial W} = X^T \frac{\partial L}{\partial Y}$ 和 $\frac{\partial L}{\partial X} =  \frac{\partial L}{\partial Y} W^T$，浮点数计算量为$2*2*bs*m*n=4*bs*m*n$，偏置项梯度 $\frac{\partial L}{\partial b}=\text{reduce_sum}(\frac{\partial L}{\partial Y}, 0)$ 可忽略不计  

    > 一般不需要计算输入$X$ 梯度  

- forward + backward = $2*3*N$，$N$表示参数量，2表示multiply+add操作
## 1
> 论文：Deep Learning Scaling is Predictable, Empirically  
> Baidu Research, 2017 Dec

### 主要内容
- this paper is the first to empirically characterize learning curve and model size scaling trends for a broad range of application domains and models.
- Our empirical results show power-law generalization error scaling across a breadth of factors, resulting in power-law exponents——the "steepness" of the learning curve——yet to be explained by theoretical work. Further, model improvements only shift the error but do not appear to affect the power-law exponent. We also show that model size scales sublinearly with data size.
- These scaling relationships have significant implications on deep learning research, practice, and systems. They can assist model debugging, setting accuracy targets, and decisions about data set growth. They can also guide computing system design and underscore the importance of continued computational scaling.
- accurately predicting generalization error scaling with training set size would provide a powerful tool for estimating the costs—in data and compute requirements—for advancing state-of-the-art (SOTA).

#### Generalization Error Scaling with Data
- learning curves measure how much training data a model family requires to reach a particular accuracy.
- generalization error curve
- loss learning curve
- Many studies theoretically predict that generalization error "learning curves" take a power-law form, $\epsilon (m) \propto \alpha m^{\beta_g}$. 
    - $\epsilon$ 为generalization error  
    - $m$ number of training samples
    - $\alpha$ is a scalar
    - $\beta_g$ scaling exponent，一般为负数, g=generation error

- Our results show that power-law learning curves exist across all tested domains. Although different applications yield different power-law exponents and intercepts
- Improved model architectures and optimizers can improve the power-law intercept, but not the exponent

#### Model Capacity Required to Fit Data
- model size scaling
- number of model parameters to fit a data set should follow $s(m) \propto  \alpha m^{\beta_p}$
    - $s(m)$ is the required model size to fit a training set of size m
    - $\alpha$ is a scalar
    - $\beta_p \in [0.5, 1]$ p=parameter

- These studies show that while model capacity might explain a model’s ability to memorize training examples, capacity may not adequately explain the model’s ability to generalize to new examples.
- Rather than reason through these complexities, it is currently easier for researchers and practitioners to over-parameterize models to fit training data
- train "hyperparameter-reduced" versions of these models on successively larger subsets (shards) of a training set to see how the accuracy of the model grows with training set size.
- "large data set" is a training set that could be reduced in size by 2-3 orders of magnitude and still be significant enough to perform valuable model architecture studies.
- Data sets: shard in steps of roughly 2x, e.g., 0.1% T, 0.2% T, 0.2% T, ...
- We use either the validation set available with training data, or if such a validation set is not available, we use a hold-out subset of T that does not overlap with any of the T shards.
- $\forall_i V \cap T_i = \phi$


![alt text](image.png)
- small data region
- power-law region
- irreducible error region

3 types of scaling limits: 

- training data is too small
- computation is too slow
- irreducible error

## 2
> 论文：Scaling Laws for Neural Language Models  
> Johns Hopkins University & OpenAI, 2020 Jan  

### 主要内容
- observe precise power-law scalings for performance as a function of training time, context length, dataset size, model size, and compute budget
- display predicted compute when using a sufficiently small batch size
- model size > batch size > steps
- PF-day $=10^{15}*24*3600=8.64*10^{19}$ FLOPs
- equation 1.7和1.8如何得来
- $B_\text{crit}$ provides a roughly optimal compromise(trade-off) between time and compute efficiency.
- $C_\text{min}$ 和 $S_\text{min}$ 分别是在极小batch_size与极大batch_size情况下的取值


## 3
> 论文：Scaling Laws for Autoregressive Generative Modeling  
> OpenAI, 2020 Oct

## Chinchilla
> 论文：Training Compute-Optimal Large Language Models  
> DeepMind, 2022 Mar
