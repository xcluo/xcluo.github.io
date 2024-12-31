### QLoRA
> 论文：QLORA: Efficient Finetuning of Quantized LLMs  
> QLoRA: **Q**uantized **Lo**w-**R**ank **A**daptation  
> University of Washington, NeurIPS 2023


#### 工作要点
QLoRA包含两部分，首先将Frozen pretrained LLM使用NF4方式量化, 随后基于LoRA思想进行PEFT，

<div class="one-image-container">
    <img src="\AI\Paper_Reading\Trick\LLM_Extend\LLM_SFT\image\qlora_diagram.jpg" style="width: 85%;">
</div>

- [x] 4-bit NormalFloat(NF4) 在符合正态分布数据上量化效果优于4-bit Integers和4-bit Floats
- [ ] double quantization双重量化
- [x] paged optimizers，GPU显存不足时，自动在GPU和CPU间进行page2page的传输，以避免OOM的现象，类似CPU内存不足时，自动在RAM和Disk间进行page2page的传输

#### 细节实现
1. 4-bit NormalFloat Quantile Quantization，此前的量化操作先验地认为模型参数在值域区间均匀分布，然而实际模型参数更倾向正态分布(如batch_norm、layer_norm、normal_initializer、truncated_normal_initializer)，因此基于正态分布密度情况量化更为合理，NF4 convert包括：

    1. 获取cdf分位数值 [`normal_map`](https://github.com/bitsandbytes-foundation/bitsandbytes/blob/main/bitsandbytes/functional.py#L291) 
        ```python
        if is_asymmetric:         # 实际使用16值: 左7等分, 右8等分
            normal_cdf_left = -norm.ppf(torch.linspace(offset, 0.5, 8)[:-1])
            normal_cdf_right = norm.ppf(torch.linspace(offset, 0.5, 9))
        
        elif not is_asymmetric:   # 实际使用15值: 左7等分, 右7等分
            normal_cdf_left = -norm.ppf(torch.linspace(offset, 0.5, 8))
            normal_cdf_right = norm.ppf(torch.linspace(offset, 0.5, 8))

        normal_cdf = normal_cdf_left + normal_cdf_right
        # 分位值归一化
        normal_map = normal_cdf/absmax(normal_cdf)
        ```
        
        > 缺省边界值 `offset=(1-1/(2*15)+1-1/(2*16))/2`  
        >  `offset≈1` 的同时要求`absmax(normal_cdf)`不过于大以**避免分位值归一化后中心大密度区域粒度模糊(归属左分位段和归属右分位段差异过小)**

    2. 归一化 `#!python x = x/absmax(x)`
    3. 获取最近分位值，`#! x = [find_nearest(v, normal_cdf) for v in x]`
    4. 映射为分位值下标对应的4-bit值

  
3. double quantization, average 0.37-bit per parameter  
        - 每64个参数共享一个量化常数  
        - 此时常规nf4下，每个块参数用于存储量化常数的额外开销为fp32/(nf4*64)=32/4/64=12.5%  
        - 双精度：对量化常数进行fp8量化，此处量化常数个数采用256  
        - 此时存储量化常数额外开销为(fp8*256+fp32)/(nf4*256*64)=8*256/4/256/64=3.125% + 0.049%=3.17%，其中fp32为量化量化常数的量化常数  
        - 每个参数的平均额外开销：fp32/64=0.5bit → (fp8*256+fp32)/64*256=0.127bit
  



```python
from scipy.stats import norm

# norm.ppf: Percent point function (inverse of `cdf`)
# norm.cdf: Cumulative distribution function
```

- https://medium.com/@levxn/lora-and-qlora-effective-methods-to-fine-tune-your-llms-in-detail-6e56a2a13f3c