- https://spaces.ac.cn/archives/9948
- https://www.cnblogs.com/mudou/p/18309199#%E7%AC%AC%E4%B8%80%E9%83%A8%E5%88%86-%E8%83%8C%E6%99%AF%E7%9F%A5%E8%AF%86%E4%BB%8E%E8%BF%9B%E5%88%B6%E8%A1%A8%E7%A4%BA%E8%B0%88%E5%88%B0%E7%9B%B4%E6%8E%A5%E5%A4%96%E6%8E%A8%E7%BA%BF%E6%80%A7%E5%86%85%E6%8F%92%E8%BF%9B%E5%88%B6%E8%BD%AC%E6%8D%A2
- https://spaces.ac.cn/archives/9948#%E8%BD%AC%E5%9C%88%E8%A7%86%E8%A7%92
- https://blog.csdn.net/v_JULY_v/article/details/135072211  
- https://www.cnblogs.com/mudou/p/18309199#%E7%AC%AC%E4%B8%89%E9%83%A8%E5%88%86-%E4%BB%8Entk-awarentk-by-parts%E5%88%B0dynamic-ntk%E6%8F%92%E5%80%BC  
- https://blog.csdn.net/z551646/article/details/140494221
### [NTK-aware Scaled RoPE]((https://www.reddit.com/r/LocalLLaMA/comments/14lz7j5/ntkaware_scaled_rope_allows_llama_models_to_have/))
NTK-aware是一种受神经正切核 (Neural Tangent Kernel) 理论启发的动态插值方案，通过非均匀维度缩放（转换表达进制），避免[直接线性内插](pi.md#基本原理)时高频维度（e.g., $m' = m/k$）的旋转角度变化过大（粒度过小，插入难度高），破坏位置编码信息的有效插入。

#### 原理：进制表达转换

$$
\begin{aligned}
    f'(x, m, \theta_i) =& f\left(x, m, h\left(\theta_i\right)\right) \\
    h(\theta_i) =& (b\lambda)^{-2*i/d} \\
    \lambda =& k ^{\frac{d}{d-2}} \\
    k =& \frac{L^{'}}{L}
\end{aligned}
$$

1. **[转换表达进制](https://spaces.ac.cn/archives/9675#%E8%BF%9B%E5%88%B6%E8%BD%AC%E6%8D%A2)**：3位十进制最多可以表示0~999，若仍想使用相同位表达更大数值范围，可转换为更大进制表示，e.g. `749==0x2ED`，此时高位至低位迭代对余数执行了 $\frac{m}{\lfloor b^{i-1}\rfloor}$ 操作
   
    > 代价是每个数值位表示范围增大，类推至位置编码即各维度需聚合的信息要求更多

2. **[RoPE与进制表达](https://spaces.ac.cn/archives/9675#%E4%BD%8D%E7%BD%AE%E7%BC%96%E7%A0%81)**，在RoPE中位置$m$编码如下，由于 `sin/cos` 为周期函数，其高位至低位可视作对位置 $m$ 的$\frac{d}{2}$位$\beta$进制化表示

    $$
    \left[ \cos\left(\frac{m}{\beta^{0}}\right), \sin\left(\frac{m}{\beta^0}\right), \cos\left(\frac{m}{\beta^{1}}\right), \sin\left(\frac{m}{\beta^1}\right), \cdots, \cos\left(\frac{m}{\beta^{d/2-1}}\right), \sin\left(\frac{m}{\beta^{d/2-1}}\right) \right]
    $$

3. **RoPE与进制转换**，实际RoPE中 $\theta_{m, i} = \frac{m}{b^{2*i/d}}, i \in \{0, 1, \dots, d/2-1\}$，转换表达进制后位置 $m$ 编码 $\theta_{m, i}' = \frac{m}{(b \lambda )^{2*i/d}}$的低、高频部分有不同表现：
    - ^^低频^^（高位）：$\theta'_{m, d/2-1} = \frac{m/\lambda^{(d-2)/d}}{b^{(d-2)/d}}= \frac{m/k}{b^{(d-2)/d}}$，对 $m$ 低频部分进行了内插操作
    - ^^高频^^（低位）：$\theta'_{m, 0} = \frac{m}{1}=m$，对 $m$ 高频部分进行了直接外推操作

    !!! info ""
        - 低、高频部分出现了差异化非均匀缩放现象，具体倾向为高频外推，低频内插，即低频部分较为宽松，适合插值，高频较为密集，位置信息插入后过于拥挤，阻碍解码
        - `context_window` 拓展倍数 $k=\lambda^{(d-2)/d}$，因此实际应用中常通过预定义拓展倍数计算 $\lambda = k^{d/(d-2)}$
        - 转换后的进制 $b' = b\lambda^{d/(d-2)}$

4. **代码实现**：[ntk_scaled_init](https://colab.research.google.com/drive/1VI2nhlyKvd5cw4-zHvAIk00cAVj2lCCC#scrollTo=b80b3f37&line=1&uniqifier=1)
   ```python
   import transformers

   old_init = transformers.models.llama.modeling_llama.LlamaRotaryEmbedding.__init__
   def ntk_scaled_init(self, dim, max_position_embeddings=2048, base=10000, device=None):
        # 拓展倍数 k
        a = 8
        # m' = k*m
        max_position_embeddings = 16384
        # 基于拓展倍数转换进制表达 lambda = k^{d/(d-2)}
        base = base * a ** (dim / (dim-2)) #Base change formula

        old_init(self, dim, max_position_embeddings, base, device)
   ```

### [Dynamic NTK](https://www.reddit.com/r/LocalLLaMA/comments/14mrgpr/dynamically_scaled_rope_further_increases/)
Dynamic NTK是NTK-aware Scaled RoPE的改进版本，基于当前输入序列长度动态调整RoPE的频率基，避免固定缩放比例带来的潜在问题，从而在短文本上保持原始性能，在长文本上实现稳定拓展。

#### 背景与动机
1. **NTK-aware Scaled RoPE局限性** 通过固定缩放倍数（e.g., $k = L_\text{test} / L_\text{train}$）调整RoPE的频率基$b\lambda$，使得模型在更长`context_window`上保持较好的性能，但该方法存在以下两个问题
    - [ ] ^^短文本性能下降^^，如固定缩放倍数 $k$ 过大，在处理短文本时旋转角可能过度压缩，影响模型对局部位置感知能力
    - [ ] ^^拓展长文本后效果突然衰减^^


#### 改进

$$
\begin{aligned}
    f'(x, m, \theta_i) =& f\left(x, m, h\left(\theta_i\right)\right) \\
    h(\theta_i) =& (b\lambda)^{-2*i/d} \\
    \lambda =& s ^{\frac{d}{d-2}} \\
    s =& \max(1, l_\text{seq}/L_\text{train})
\end{aligned}
$$

1. **动态调整缩放倍数**，基于每次输入序列长度动态更新缩放倍数 $k=\max(1, l_\text{seq}/L_\text{train})$，防止模型在短文本场景下$l_\text{seq} \le L_\text{train}$出现性能折扣，长文本场景下$l_\text{seq} \ge L_\text{train}$效果突然衰减
    - $l \le L_{train}$，保持原有RoPE不变
    - $l \ge L_{train}$，使用NTK-aware Scaled RoPE 进行 `context_window` 拓展


### [NTK-by-parts](https://github.com/jquesnelle/yarn/pull/1) 

NTK-by-Parts 是 NTK-aware Scaled RoPE 的进一步优化方法，通过对不同频率区间差异化缩放（类似于[NTK-RoPE-fixed](https://spaces.ac.cn/archives/9706/comment-page-1#%E6%B7%B7%E5%90%88%E8%BF%9B%E5%88%B6)），解决长序列拓展时的高频震荡问题。


#### 背景与动机
1. **NTK-aware 局限性**，对所有频率维度采用统一缩放策略，可能导致
    - ^^高频信息过度压缩^^，旋转角度变化过快
    - ^^低频信息缩放不足^^，难以捕获超长程以来
2. **Dynamic NTK局限性**，未区分不同频率区间，可能无法最优处理超长序列。

3. **RoPE角速度与波长**，在RoPE位置编码中，特征维度 $i$ 的角速度 $\theta_i = \frac{1}{b^{2*i/d}}$，此时该角速度下可被训练的点数（即波长）为$\lambda_i = \frac{2\pi}{\theta_i} = 2\pi {b^{2*i/d}}$，不同特征维度波长特性不同：

    - 低维特征波长较小，充分训练对应所需最小相对长度较小
    - 高维特征波长较大，充分训练对应所需最小相对长度较大

    > - 相对长度极端情况下为$L_\text{train}-1$
    > - 应用PI线性插值后，最小相对长度增为 $k=L'/L$ 倍
    
#### 改进
基于上述不同特征维度波长和充分训练所需最小相对长度的差异性，从以下进行了改进：

1. **将RoPE的频率维度划分为多个区间**，如低频、中频、高频
2. **对不同频率区间差异化缩放**，重置各维度角速度 $h(\theta_i)$
    - ^^高频-大角速度-小波长^^，已充分训练，弱缩放或保持原样直接外推
    - ^^低频-小角速度-大波长^^，部分训练，内插强缩放
    - ^^中频-中角速度-中波长^^，trade-off
3. **避免频率区间冲突**，保持位置编码的连续性，避免缩放后差异过大

    $$
    \begin{aligned}
        f'(x, m, \theta_i) =& f\left(x, m, h\left(\theta_i\right)\right) \\
        \gamma(r) =& \begin{cases}
            0 & \text{if } r \lt \alpha \\
            1 & \text{if } r \gt \beta \\
            \frac{r-\alpha}{\beta - \alpha} & \text{otherwise} \\
        \end{cases} \\
        h(\theta_i) =& \left(1 - \gamma\left(r_i \right)\right)\frac{\theta_i}{k} + \gamma\left(r_i \right) \theta_i \\
        r_i =& \frac{L_\text{train}}{\lambda_i} = \frac{L_\text{train}}{2\pi {b^{2*i/d}}}
    \end{aligned}
    $$

    > - $r_i$ 表示训练长度与特征维度 $i$ 的比率
    > - $k=L'/L$ 为拓展倍数，表现为线性插值比例，也常用 scale factor $s$ 表示
    > - $\alpha=1, \beta=32$



### YaRN
