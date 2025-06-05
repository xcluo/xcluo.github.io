### QLoRA
> 论文：QLORA: Efficient Finetuning of Quantized LLMs  
> QLoRA: **Q**uantized **Lo**w-**R**ank **A**daptation  
> University of Washington, NeurIPS 2023


#### 工作要点
QLoRA包含两部分，首先将Frozen pretrained LLM使用NF4方式量化, 随后基于LoRA思想进行PEFT，

$$
\begin{aligned}
Y^{\text{BF16}} = X^{\text{BF16}}\text{doubleDequant}(c_1^{\text{FP32}}, &c_2^{\text{k-bit}}, W^{\text{NF4}}) + X^{\text{BF16}}A^{\text{BF16}}B^{\text{BF16}} \\
\text{doubleDequant}(c_1^{\text{FP32}}, c_2^{\text{k-bit}}, W^{\text{NF4}})&=\text{dequant}(\text{dequant}(c_1^{\text{FP32}}, c_2^{\text{k-bit}}), W^{\text{NF4}}) \\
&= W^{\text{BF16}}
\end{aligned}
$$

<div class="one-image-container">
    <img src="\AI\Paper_Reading\Trick\LLM_Extend\LLM_SFT\image\qlora_diagram.jpg" style="width: 85%;">
</div>

- [x] 4-bit NormalFloat(NF4) 在符合正态分布数据上量化效果优于4-bit Integers和4-bit Floats
- [x] double quantization双重量化，进一步都量化缩放常数进行缩放以减少存储开销
- [x] paged optimizers，GPU显存不足时，自动在GPU和CPU RAM间进行page2page的传输，以避免OOM的现象，类似CPU内存不足时，自动在RAM和Disk间进行page2page的传输
- [x] finetue 33B/65B(>780GB) LLM of GPU memory to 24GB/48GB without sacrificing performance

#### 细节实现
1. **4-bit NormalFloat Quantile Quantization**，此前的量化操作先验地认为模型参数在值域区间均匀分布，然而实际模型参数更倾向正态分布(如batch_norm、layer_norm、normal_initializer、truncated_normal_initializer)，因此基于正态分布密度情况量化更为合理，NF4 convert包括：

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
        # [-1.0, -0.6961928009986877, -0.5250730514526367, -0.39491748809814453, -0.28444138169288635, -0.18477343022823334, -0.09105003625154495, 0.0, 0.07958029955625534, 0.16093020141124725, 0.24611230194568634, 0.33791524171829224, 0.44070982933044434, 0.5626170039176941, 0.7229568362236023, 1.0]
        ```
        
        > 缺省cdf边界值 `offset=(1-1/(2*15)+1-1/(2*16))/2`  
        >  `offset≈1` 的同时要求`absmax(normal_cdf)`不过于大以**避免分位值归一化后中心大密度区域粒度模糊(归属左分位段和归属右分位段差异过小)**

    2. 归一化 `#!python x = x/absmax(x)`
    3. 获取最近分位值，`#! x = [find_nearest(v, normal_cdf) for v in x]`
    4. 映射为分位值下标对应的4-bit值

  
2. **double quantization**，对量化常数进一步缩放(二阶量化常数共享一阶量化常数)以减少存储开销，假设双重量化block size分别为$B_1=64, B_2=256$
    
    - 无双重量化，一阶量化常数类型为fp32
        - 平均每个参数额外开销为fp32/64=0.5bit；
        - 存储量化常数额外开销比例为fp32/(64\*nf4)=12.5%
    - 应用双重量化，一阶、二阶量化常数类型分别为fp32和fp8
        - 平均每个参数额外开销为(256\*fp8 + fp32)/(256\*64)=0.127bit
        - 存储量化常数额外开销比例为(256\*fp8 + fp32)/(256\*64\*nf4)=3.175%
  

#### 实验效果
1. NF4 is better other 4-bit quantization
    <div class="one-image-container">
        <img src="\AI\Paper_Reading\Trick\LLM_Extend\LLM_SFT\image\qlora_vary_data-type.png" style="width: 60%;">
    </div>
2. Guanaco is the best-performing model after GPT-4
    <div class="one-image-container">
        <img src="\AI\Paper_Reading\Trick\LLM_Extend\LLM_SFT\image\qlora_guanaco_performance.png" style="width: 85%;">
    </div>
3. NF4 quantization enables inference speedup
    <div class="one-image-container">
        <img src="\AI\Paper_Reading\Trick\LLM_Extend\LLM_SFT\image\qlora_inference_speedup.png" style="width: 85%;">
    </div>
```python
from scipy.stats import norm

# norm.ppf: Percent point function (inverse of `cdf`)
# norm.cdf: Cumulative distribution function
```



- https://medium.com/@levxn/lora-and-qlora-effective-methods-to-fine-tune-your-llms-in-detail-6e56a2a13f3c