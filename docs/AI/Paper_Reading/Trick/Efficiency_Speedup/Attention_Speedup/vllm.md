## vLLM
> 论文：Efficient Memory Management for Large Language Model Serving with PagedAttention  
> vLLM: **v**irtual **L**arge **L**anguage **M**odel inference engine  
> Github：[vllm](https://github.com/vllm-project/vllm)  
> UC Berkeley & Stanford University & Independent Researcher & UC San Diego 2023 Sep, SIGOPS 2023

### 主要内容
vLLM是一个开源的推理引擎，专**为LLM的推理和部署而设计**，核心特性包括高效的服务吞吐量、优化的内存管理和多种解码算法支持。

- [x] 解决内存碎片化问题
- [x] 实现并行采样或beam search生成的多个输出序列间部分KV缓存可能共享

https://blog.csdn.net/m0_59164520/article/details/141869967

#### PagedAttention
核心思想是将KV cache组织为固定大小块(block)中类似虚拟内存中的"页"。关键技术特点如下：

1. **非连续存储**：允许将连续的K和V存储在不连续的物理内存中（通过维护block table映射），每个KV块包含固定数量token的KV向量  
2. **块式注意力计算**：注意力计算转换为对块的计算形式，PagedAttention内核能够分别识别和获取不同KV块进行计算  
3. **灵活的内存管理**：可将块视为页、token视为字节、请求视为进程，实现类似操作系统的内存管理

#### 内存共享
1. prompt部分KV cache共享  
2. 其余非相关部分独立存储