#### 多模态融合
- [token](Multimodality_Fusion/token_modality.md)
- [sound](Multimodality_Fusion/sound_modality.md)
- [shape](Multimodality_Fusion/shape_modality.md)
- [fusion](Multimodality_Fusion/modality_fusion.md)

#### 多任务学习
- multitask learning


#### 集成学习
- Boosting
      - [LightGBM](Ensemble/Ensemble/Boosting/lightgbm.md)
- Bagging
- Staking
- MoE
- [dropout](AutoEncoder/Dropout/dropout.md)
- [DAE](AutoEncoder/DAE/dae.md)
- [VAE](AutoEncoder/VAE/vae.md)

#### 大模型使用
- [蒸馏](LLM_Extend/distillation/distillation.md)
- [LLM_SFT](LLM_Extend/LLM_SFT/LLM_SFT.md)

#### 存储优化
- Attention优化
    - [flash attention](Memory_Saving/Flash_Attention/FlashAttention.md)
    - [MQA](Memory_Saving/Attention_Variants/mqa/#mqa)、[GQA](Memory_Saving/Attention_Variants/mga/#gqa)
- [Parallelism](Memory_Saving/Parallelism/parallelism.md)
    - TP(tensor parallelism)
    - PP(pipeline parallelism)
    - CP(context parallelism)
    - DP(data parallelism)
- 量化：[Quantization](Memory_Saving/Quantization/quantization.md)，减少前向过程中存储消耗，但反向过程则不会有该特性