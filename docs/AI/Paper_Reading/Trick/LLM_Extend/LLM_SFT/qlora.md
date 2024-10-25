Quantized Low-Rank Adaptation

1. 4-bit normalfloat, better than b-bit integers and 4-bti floats  
        - offset = (1 - 1/(2*15) + 1 - 1/(2*16))/2，offset的值理论上越接近1越好，但只要不是太过不合理问题都不大（类似于sigmoid超过一定值分位图占比过小）  
        - 累计密度分位划分，分位点除以边界值归一化为[-1, 1]
  
2. double quantization, average 0.37-bit per parameter  
        - 每64个参数共享一个量化常数  
        - 此时常规nf4下，每个块参数用于存储量化常数的额外开销为fp32/(nf4*64)=32/4/64=12.5%  
        - 双精度：对量化常数进行fp8量化，此处量化常数个数采用256  
        - 此时存储量化常数额外开销为(fp8*256+fp32)/(nf4*256*64)=8*256/4/256/64=3.125% + 0.049%=3.17%，其中fp32为量化量化常数的量化常数  
        - 每个参数的平均额外开销：fp32/64=0.5bit → (fp8*256+fp32)/64*256=0.127bit
  
3. paged optimizers, using NVIDIA unified memory to avoid the gradient checkpointing memory spikes
        - GPU显存不足时，自动在GPU和CPU间进行page2page的传输，类似CPU内存不足时，自动在RAM和Disk间进行page2page的传输，以避免OOM的现象

```python
from scipy.stats import norm

# norm.ppf: Percent point function (inverse of `cdf`)
# norm.cdf: Cumulative distribution function
```

- https://medium.com/@levxn/lora-and-qlora-effective-methods-to-fine-tune-your-llms-in-detail-6e56a2a13f3c