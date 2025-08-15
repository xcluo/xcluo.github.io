## KVQuant
> 论文：KVQuant: Towards 10 million context length LLM inference with KV cache quantization  
> Github：[KVQuant](https://github.com/SqueezeAILab/KVQuant)  
> UC Berkeley & Stanford University & Independent Researcher & UC San Diego, 2023 Sep, SIGOPS 2023

### 主要内容
- Per-Channel (Group-wise) Pre-RoPE Key Activations Quantization, Keys exhibit outliers in specific channels before applying RoPE, However, the outlier channel magnitudes become less consistent after applying RoPE
    - per-channel quantization provides significant accuracy benefits for Keys but not for Values.
    - 单纯的对pre-RoPE Key进行quantization会导致重计算，per-channel quantization can also be challenging due to the need to recompute scaling factors as tokens are added to the Key cache. We show that we can calibrate offline for scaling factors, thereby avoiding expensive online recomputation
    - Key vectors after applying this rotation is that it leads to mixing pairs of channels by different amounts for different positions in the sequence
- ![alt text](image-3.png)
- per-tensor Non-Uniform KV Cache Quantization offline on calibration data using a k-means
solver
    - Uniform quantization is suboptimal for KV cache quantization since the Query and Key activations are non-uniform
    - Fisher information matrix，梯度平方矩阵
    - $Q(A) = \argmin_{Q} \sum_{i=1}^N \mathcal{F}_{ii}(A_i - Q(A_i))^2$
    - Appendix L shows how computing the required Fisher information for the LLaMA-65B model takes only a few minutes
- Per-Vector Dense-and-Sparse Quantization, where we isolate outliers separately for each vector to minimize skews in quantization ranges.
    - the majority of elements are contained within a small percentage of the dynamic range, isolate a small percentage of numerical outliers, we can restrict the range that we need to represent, thereby allowing us to represent the remaining elements with greater precision (通过上述公式offline矫正)
    - use a different outlier threshold per-vector(per-dimension) (either a separate threshold per-channel for per-channel quantization)
    - Appendix J will demonstrate the benefits of removing a small percentage of numerical outliers and keeping them in full precision, as well as the advantages of per-vector dense-and-sparse quantization over using a single global outlier threshold for each layer
- per-vector outlier detection outperforms per-matrix outlier
- 3.5 Attention Sink-Aware Quantization