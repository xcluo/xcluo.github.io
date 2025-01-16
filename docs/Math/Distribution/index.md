### 分布
- normal distribution正态分布
$X \sim\mathcal{N}(\mu, \sigma^2)$，密度函数$f(x)=\frac{1}{\sqrt{2\pi}\sigma}e^{-\frac{(x-\mu)^2}{2\sigma^2}}$

- poisson distribution泊松分布，是一种离散概率分布，用于描述在一定时间或空间内事件发生的次数。它特别适用于当这些事件是独立发生且以恒定的平均速率出现时的情景。


### 采样方式
#### Inverse Sampling
- 逆采样inverse sampling是一种可以从概率分布中生成随机样本的计数。特别**适用于离散分布或某些特定的连续分布**，其中累计分布函数CDF是已知的，并且可以方便求逆，逆采样的基本思路是利用分布的累计分布函数CDF来生成随机数，步骤为
    1. 确定累计分布函数$F(x)$  
    2. 从均匀随机分布`uniform(0, 1)`中生成随机数$u$
    3. 使用逆CDF函数$F^{-1}(x)$满足$x=F^{-1}(u)$【此处可以理解x为$F(x_i)\ge u$的最小$x_i$值】，此时x即为生成的随机数样本


#### Rejection Sampling
- 拒绝采样，也叫接受-拒绝采样Acceptance-Rejection Sampling是一种用于从复杂分布中生成样本的技术。当目标分布$P(x)$的**概率密度函数PDF难以直接采样**时，但可以计算其值或比例，那么就可以使用**拒绝采样方法来间接地生成**样本。步骤为：
    1. 选择一个容易采样的proposal distribution提议分布$Q(x)$，要求提议分布$Q(x)$需要覆盖目标分布$P(x)$的定义域X  
    2. 找到一个常数c，使得对于所有x，$cQ(x)\ge P(x)$，保证分布$P(x)$的值域总是在$cQ(x)$之下  
    3. 采样和接受/拒绝
          - 依照提议分布Q中采样得到x_0  
          - 从均匀分布`uniform(0, 1)` 中采样得到 $u$
          - 如果$u\lt\frac{P(x_0)}{cQ(x_0)}$，则接受x_0作为P生成的样本；否则拒绝x_0，重复上述过程直到获得足够多的样本  
    > 常数c需要手动选择并满足条件2

> https://www.zhihu.com/question/38056285/answer/1803920100

- 重要性采样importance sampling

- 马尔科夫链的当前状态只依赖于前一状态，即$P(x_t\vert x_1, x_2, \dots, x_{t-1}) = P(x_t\vert x_{t-1})$
    - $\pi$ 为马尔科夫链的平稳分布，满足$\pi P = \pi$
- 基于马氏链的蒙特卡罗方法采样MCMC (Markov Chain & Monte Carlo)
    - Detailed Balance细致平衡$\pi(i)P(i, j)=\pi(j)P(j, i)  \rightarrow \pi(i)Q(i,j)\alpha(i, j)=\pi(j)Q(j,i)\alpha(i, j)$
    - $P(i, j)=Q(i, j)\alpha(i, j) \rightarrow \alpha(i, j)=\min\{\frac{\pi(j)Q(j,i)}{\pi(i)Q(i,j)}, 1\}$
    - 其中Q为状态转移矩阵，也可以是状态转移概率密度函数，$P(A\rightarrow B) = P(B\vert A)$
    - 经过1~t时间burn-in燃烧期的状态变换后，在[t+1, ∞]后进入细致平衡期，进入细致平衡期后开始进行最终采样
    - 采样点间不互相独立，mixing time t可能会很长
- MH采样 Metropolis Hastings
- Gibbs sampling吉布斯采样
    - 联合概率分布转化为条件概率（高维转化为低维），不一定要轮换坐标轴，只需要符合条件概率分布进行采样即可，不拒绝，所有采样均接受
    - 是α=1情况下的MH采样，为MH采样的特殊形式。适用于随机变量X维度非常高的情况，从t到t+1时刻，只改变一个维度的值。状态转移矩阵取得就是目标概率p(X)。

> https://www.cnblogs.com/pinard/p/6645766.html  
> [逆采样、拒绝采样、MH采样、MCMC采样](https://www.bilibili.com/video/BV1ey4y1t7Jb/?spm_id_from=333.337.search-card.all.click&vd_source=782e4c31fc5e63b7cb705fa371eeeb78)
> https://www.zhihu.com/topic/20683707/top-answers
> https://www.cnblogs.com/feynmania/p/13420194.html
> https://www.zhihu.com/question/38056285/answer/1803920100
>
> https://zhuanlan.zhihu.com/p/669645171