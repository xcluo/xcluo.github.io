- [ntk-aware](https://www.reddit.com/r/LocalLLaMA/comments/14lz7j5/ntkaware_scaled_rope_allows_llama_models_to_have/)  
- [by parts](https://github.com/jquesnelle/yarn/pull/1)  
- [dynamic](https://www.reddit.com/r/LocalLLaMA/comments/14mrgpr/dynamically_scaled_rope_further_increases/)
- https://spaces.ac.cn/archives/9675
- https://spaces.ac.cn/archives/9706
- https://spaces.ac.cn/archives/9948
- https://blog.csdn.net/v_JULY_v/article/details/135072211  
- https://www.cnblogs.com/mudou/p/18309199#%E7%AC%AC%E4%B8%89%E9%83%A8%E5%88%86-%E4%BB%8Entk-awarentk-by-parts%E5%88%B0dynamic-ntk%E6%8F%92%E5%80%BC  
- https://zhuanlan.zhihu.com/p/695978857
- https://blog.csdn.net/z551646/article/details/140494221
## NTK-aware Scaled RoPE
NTK-aware是一种基于神经正切核K(Neural Tangent Kernel)理论的模型优化方案，核心思想是利用 NTK 理论性质，调整模型的参数更新或特征表示，优化其在插值任务（如分类、回归）中的表现

### 原理：进制转化
- 思路：动态调整频率base，因为进制转换（10→16），`749 → 2,14(E),13(D)`，使用相同的位表示更长的范围（代价为每位的数字从0~9变成了0~15）
- 还可通过设置拓展上界，缩窄每位的表示范围，如3位数表示2000，$x^3 + x^2 + x^1 \ge 2000, x\ge 13$
- $\theta_{m, i} = m * b^{-2(i-1)/d}, j\in \{1, 2, \cdots, d/2\}$，$b$ 为基数，每个位置的高维度三角函周期越来越大，频率越来越低；
- 低维高频周期小：高频插值后相对密集，为不导致拥挤，目标为不进行缩放，退化为原始RoPE，保留短距离位置关系（$L \lt L_{train}$）
- 高维低频周期大：低频插值后依然相对宽松，缩放无明显影响目标，提升外推效果（$L \gt L_{train}$）  
- 核心思想：并非如PI一样针对所有维度平均缩放，而是低维高频不缩放、高维低频才缩放
- 假定外推缩放倍数为 $k$，base缩放因子为 $\lambda$，目标符合 $i \rightarrow 0, f(\lambda, i) \rightarrow 1$，$i \rightarrow d/2, f(\lambda, i) \approx k^{-1}$ 
- 低频目标代入求得 $m*(b\lambda)^{(-d+2)/d}= \frac{m}{k}b^{(-d+2)/d}$，$\lambda = k^{d/(d-2)}, k = \lambda ^{(d-2)/d}$
- 因此 $\theta_{i}^{'} = m*{(b\lambda)^{-2(i-1)/d}} = m*{b^{-2(i-1)/d}*k^{-2(i-1)/(d-2)}}$

### NTK-by-parts

通过分段差异化缩放旋转频率，解决长序列外推时的高频震荡问题。

$$
\gamma(r) = \begin{cases}
    0 & \text{if } r \lt \alpha \\
    \frac{r-\alpha}{\beta - \alpha} & \text{otherwise} \\
    1 & \text{if } i \gt \beta \\
\end{cases}
$$

- 比较维度的波长，当波长足够长时（比如大于L），进行插值；当波长较小时，更倾向于原始RoPE保持不变
- 维度越低，频率越大，波长越短，比值越大，更倾向于原始RoPE保持不变
- 维度越高，频率越小，波长越大，比值越小，更倾向插值
- $h(\theta_d) = \big(1 - \gamma(r(d))\big)\frac{\theta_d}{s} + \gamma\big(r(d)\big)\theta_d$
- $r(i) = \frac{L_{train}}{\lambda_i}$，训练上下文大小和指定维度波长$2\pi b^{2(i-1)/d}$的比值

插值步骤

1. 初始化RoPE的频率 $\theta_b$
2. 根据公式通过加权组合计算 $h(\theta_d)$，对每个维度的频率进行调整，主要涉及 $\frac{\theta_{d}}{s}$ 和原始频率 $\theta_d$

### Dynamic NTK
- 每次前向传递中，更新缩放倍数 $k=\max(1, l/L)$，其中$l$表示当前序列的序列长度，防止模型在长度小于$L$时出现性能折扣，大于$L^{'}$时出现退化

1. $k=\max(1, l/L), b = b*k^{d/(d-2)}$
2. 当$l \gt L_{train}$，进行插值外推
3. 当$l \lt L_{train}$，保持原有RoPE不变


$s = (s* l/L_{train}) - (s-1)$