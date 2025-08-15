## ZeroQuant
> 论文：ZeroQuant: Efficient and Affordable Post-Training Quantization for Large-Scale Transformers  
> Microsoft, 2022 Jun, NeurIPS 2022

### 主要内容


- PTQ类型 targeting both INT8 and INT4/INT8 mixed-precision quantization
- a fine-grained hardware-friendly quantization scheme for both weight and activations
    - groupwise quantization for weight and token-wise quantization for activations
- a novel affordable layer-by-layer knowledge distillation algorithm (LKD) even without the access to the original training data
    - for INT4/INT8 mixed-precision quantization
- a highly-optimized quantization system backend support to remove the quantization/dequantization overhead
- reduce the precision for weights and activations from FP16 to INT8 in a cost-free with minimal accuracy impact
#### Fine-grained Hardware-friendly Quantization Scheme(ZeroQuant)
- WNAM，表示weight以及activation的量化位数
- Block-wise Quantization
- Group-wise Quantization for Weights, 将矩阵拆分成多组分别进行量化
- Token-wise Quantization for Activations，对于每个token $h_t \in \mathbb{R}^d$，计算其最绝对值最大值$\alpha_t$再计算其缩放因子 $s_t = \frac{2^{b-1} - 1}{\alpha_t}$


#### LKD
KD的缺陷

- KD needs to hold a teacher and a student model together during the training, which dramatically increases the memory and compute cost
- KD usually requires full training of the student model. Therefore, several copies (gradient, first/second order momentum) of the weight parameters need to be stored in memory to update the model
- KD generally requires original training data, which sometimes are not accessible due to privacy/confidential issues


layer-by-layer distillation (LKD) algorithm，原始层$L_k$，量化层$\hat{L}_{k}$

- 使用上一层输出作为输入 $h^{k-1}$，通过$\mathcal{L}_{\text{LKD}, k}=\text{MSE}(L_k(h^{k-1}) - \hat{L}_k(h^{k-1}))$ 优化量化层$\hat{L}_{k}$，MSE也可替换为其它loss，如KL divergence  
-LKD does not rely on the original training data
- 可对量化后的weight进行优化调整

#### Quantization-Optimized Transformer Kernels
- 使用 CUTLASS(CUDA Templates for Linear Algebra Subroutines) INT8 GeMM进行算子融合以较少加载和数据移动的开销
