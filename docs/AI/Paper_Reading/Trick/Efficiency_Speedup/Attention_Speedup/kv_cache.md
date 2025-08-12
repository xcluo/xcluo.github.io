- [x] kv cache：https://juejin.cn/post/7362789570217885759

## KV Cache
应用 KV Cache 优化的生成大模型在推理过程包含了两个阶段：

1. **Prompt预填充**：一次性将Prompt Input（Full Attention）输入至LLM中生成每层的 KV Cache
2. **解码**：时序更新并使用KV Cache，执行`next_token_prediction`（Causal Attention）操作

### 解码阶段
KV cache 将各层 $k^n_{1:t-1}$ 与 $v^n_{1:t-1}$，其中per-head缓存 `K/V.shape = (seq_len, head_dim)`，推理解码时per-head基于$h_t^n$的$q_t^n, k_t^n, v_t^n$ 计算以及缓存读取、更新后执行 

$$
\text{Attention}(q^n_t, K^n_{1:t}, V^n_{1:t}) = \text{Softmax}\left(\frac{\left(q_t^n\right)^T \left(K^n_{1:t}\right)^T}{\sqrt{d_h}}\right) V^n_{1:t}
$$

> 只需要基于 $q^n_t$ 与 $K^n_{1:t}$ 便可计算1~t的注意力权重向量，无需缓存或重计算历史$q^n_{\lt t}$
### FLOPs分析

操作项 | w/o KV Cache| w/ KV Cache
:---: | :---: | :---:
$Q, K, V$ | $3*2*ld^2$ |  
$QK^T$ |
$\text{Softmax}(logit)$ |
$score\cdot V$ |
$O$ |
Attention | 
FFN |

- kv cache 量化
- [x] Attention softmax后除以$\sqrt{d_h}$是因为权重矩阵中每个元素都是通过两个(1， d_h)方差为1的向量相乘得到的，基于正态分布累加后的标准差公式可知该值方差变为$\sqrt{d_h}$，因此执行该操作，不除以$\sqrt{d_h}$，根据softmax函数曲线，softmax结果表现更倾向于one-hot分布，[会带来梯度消失问题](https://spaces.ac.cn/archives/8620/comment-page-4#comment-24076)

- truncated normal的基于正态分布 $\mathcal{N}(\mu, \sigma^2)$，对于在$[u-2\sigma, u+2\sigma]$范围内采样结果保留，其均值为$\mu$，方差为

    $$
    \gamma = \frac{\int_{-2}^2 e^{-x^2/2}x^2 dx}{\int_{-2}^2 e^{-x^2/2} dx} = 0.7737413
    $$

- 若要得到方差为$\sigma^2$ 采样结果，需要对传入的标准差执行 $\sigma *= \frac{1}{\sqrt{\gamma}} = 1.1368472\sigma$
- https://spaces.ac.cn/archives/8620
- https://spaces.ac.cn/archives/8823