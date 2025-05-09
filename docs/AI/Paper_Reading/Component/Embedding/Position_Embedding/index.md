## Position Embedding

### Absolute Position Embedding

### Relative Position Embedding
- [旋转位置编码RoPE](RelativePE/RoPE.md)

### Position Embedding Patches

#### 模型序列长度拓展 Extend Context Window
1. 外推法(Extropolation)：短文本训练，长文本应用
    - [ALiBi](PE_patch/alibi.md)
    - [LeX](PE_patch/lex.md)
2. 插值法(Interpolation)：直接长文本二次训练
    - [Position Interpolation](PE_patch/pi.md)
    - [NTK-Aware]
    - [YaRN](PE_patch/yarn.md)