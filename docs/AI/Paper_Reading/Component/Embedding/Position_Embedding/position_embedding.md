### Absolute Position Embedding

### Relative Position Embedding
- [旋转位置编码RoPE](RelativePE/rope.md)

### Position Embedding Patches

#### 模型序列长度拓展 Extend Context Window
1. **外推**(Extrapolation)：短文本训练，长文本应用
    - [ALiBi](PE_patch/alibi.md)
    - [LeX](PE_patch/lex.md)
2. **内插**(Interpolation)：长文本二次训练或直接应用
    - 位置维度线性缩放：[Position Interpolation](PE_patch/pi.md)
    - 特征频率维度非均匀缩放：[NTK-Aware](PE_patch/ntk-aware.md)，[YaRN](PE_patch/yarn.md)