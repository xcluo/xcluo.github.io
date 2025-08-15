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
$Q, K, V$ | $3*2*ld^2$ |  $3*2*d^2$
$QK^T$ | $2*l^2d$ | $2*ld$
$score\cdot V$ | $2*l^2d$ | $2*ld$
$O$ | $2*ld^2$ | $2*d^2$
Attention | $l*(8d^2 + 4ld)$ | $8d^2 + 4ld$
FFN | $2*2*ldd_{ff}$ | $2*2*dd_{ff}$
block | $l*(24d^2 + 4ld)$ | $24d^2 + 4ld$


> - FLPOs矩阵操作包括multiply + add两部分，因此需要乘以2
> - 通常情况下$d_{ff}=4d$


### 量化

