## ZeRO
> 论文：ZeRO: Memory Optimizations Toward Training Trillion Parameter Models  
> ZeRO：**Ze**ro **R**edundancy **O**ptimizer  
> Github：[DeepSpeed](https://github.com/deepspeedai/DeepSpeed)

DeepSpeed 是由微软开发的一个开源深度学习优化库，旨在提高大规模模型训练的效率和可扩展性。

### 主要内容
in mixed precision training:

- fp16 weights
- fp16 gradients
- fp32 momentum, fp32 variance and fp32 copy of weights
- fp16 activations

in fp32 precision training:

- fp32 weights
- fp16 gradients
- fp32 momentum, fp32 variance
- fp32 activations