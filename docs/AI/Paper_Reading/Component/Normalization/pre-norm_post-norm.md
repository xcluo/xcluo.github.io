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

在Post-Norm中，越底层的Layer占比越小，残差“名存实亡”

- 随着$l$的增长，模型训练不稳定性增加，收敛慢；
- Post-Norm模型的训练极度依赖warmup

### [Pre-Norm](https://spaces.ac.cn/archives/8620#%E6%AE%8B%E5%B7%AE%E8%BF%9E%E6%8E%A5)
即在残差连接操作前执行Norm操作，如下式：


$$
\begin{aligned}
    x_{t+1} =& x_t + F_t\big(\text{Norm}(x_t)\big)\\
    \text{可理解为 } x_{t+1} =& \frac{x_{t} + F_{t}(x_t)}{\sqrt{2}} \\
    x_{l} =& x_{l-1} + F_{l-1}\big(\text{Norm}(x_{l-1})\big) \\
    =& x_{l-2} + F_{l-2}\big(\text{Norm}(x_{l-2})\big) + F_{l-1}\big(\text{Norm}(x_{l-1})\big) \\
    =& \cdots \\
    =& x_{0} + F_{0}\big(\text{Norm}(x_{0})\big) + F_{1}\big(\text{Norm}(x_{1})\big) + \cdots  + F_{l-1}\big(\text{Norm}(x_{l-1})\big)
\end{aligned}
$$

