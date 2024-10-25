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
- prefix tuning
- Instrument Finetune
- Prompt Finetune
- Transformer MoE
- Exact Match, F1 score

#### 存储优化
- Attention优化
    - [flash attention](Memory_Saving/Flash_Attention/FlashAttention.md)
    - [MQA](Memory_Saving/Attention_Variants/mqa/#mqa)、[GQA](Memory_Saving/Attention_Variants/mga/#gqa)
    - KV cache https://blog.csdn.net/ningyanggege/article/details/134564203
- [Parallelism](Memory_Saving/Parallelism/parallelism.md)
    - TP(tensor parallelism)
    - PP(pipeline parallelism)
    - CP(context parallelism)
    - DP(data parallelism)
- 量化：[Quantization](Memory_Saving/Quantization/quantization.md)，减少前向过程中存储消耗，但反向过程则不会有该特性


目标重要性衡量  

    - 比较去除目标的前后变化，一般差异值越大表示目标越重要  
    - AdaLoRA奇异值重要性衡量(equation 11)