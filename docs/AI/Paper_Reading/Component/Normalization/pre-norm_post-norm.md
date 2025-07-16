### [Post-Norm](https://spaces.ac.cn/archives/8620#%E6%AE%8B%E5%B7%AE%E8%BF%9E%E6%8E%A5)
即在残差连接操作后执行Norm操作，如下式：

$$
\begin{aligned}
    x_{t+1} =& \text{Norm}\big(x_t + F_t(x_t)\big) \\
    \text{可理解为 } x_{t+1} =& \frac{x_{t} + F_{t}(x_t)}{\sqrt{2}} \\
    \Rightarrow x_l =& \frac{x_{l-1}}{\sqrt{2}} + \frac{F_{l-1}(x_{l-1})}{\sqrt{2}} \\
    =& \frac{x_{l-2}}{2} + \frac{F_{l-2}(x_{l-2})}{2} + \frac{F_{l-1}(x_{l-1})}{\sqrt{2}} \\
    =& \cdots \\
    =& \frac{x_{0}}{2^{l/2}} + \frac{F_{0}(x_{0})}{2^{l/2}} + \frac{F_{1}(x_{1})}{2^{(l-1)/2}} + \frac{F_{2}(x_{2})}{2^{(l-2)/2}} + \cdots + \frac{F_{l-1}(x_{l-1})}{2^{1/2}}
\end{aligned}
$$

在Post-Norm中，越低层的Layer残差权重占比越小，其残差“名存实亡”

- 随着$l$的增长，模型（低层）训练不稳定性增加，收敛慢；
- Post-Norm模型的训练极度依赖warmup

### [Pre-Norm](https://spaces.ac.cn/archives/8620#%E6%AE%8B%E5%B7%AE%E8%BF%9E%E6%8E%A5)
即在残差连接操作前执行Norm操作，如下式：


$$
\begin{aligned}
    x_{t+1} =& x_t + F_t\big(\text{Norm}(x_t)\big)\\
    =& x_{t-1} + F_{t-1}\big(\text{Norm}(x_{t-1})\big) + F_{t}\big(\text{Norm}(x_{t})\big) \\
    =& \cdots \\
    =& x_{0} +\underbrace{F_{0}\big(\text{Norm}(x_{0})\big) + F_{1}\big(\text{Norm}(x_{1})\big) + \cdots  + F_{t}\big(\text{Norm}(x_{t})\big)}_{t+1} \\
    \text{由上式可知 }x_{t+1}&\text{ 是增量模型，后t+1项为同一数量级，于是有}\text{Norm}(x_t) = \frac{x_t}{\sqrt{t+1}} \\
    \Rightarrow x_l =& x_0 + F_{0}(x_{0}) + F_{1}\Big(\frac{x_1}{\sqrt{2}}\Big) + \cdots + F_t\Big(\frac{x_{l-1}}{\sqrt{l}}\Big) \\
\end{aligned}
$$

在Pre-Norm中，各层Layer残差通道是平权的，更适合训练多层模型

- 残差作用较 [Post-Norm](#post-norm) 更加明显，因此更好优化模型；
- 随着$l$的增长，$x_l$方差将会很大，所以在接预测层之前$x_l$也还要加个Norm操作  
- 在Embedding层后添加norm再输入，有利于提升训练稳定性，但该Pre-Norm操作可能会带来一定的性能损失