### 多模态融合
- [token](Multimodality_Fusion/token_modality.md)
- [sound](Multimodality_Fusion/sound_modality.md)
- [shape](Multimodality_Fusion/shape_modality.md)
- [fusion](Multimodality_Fusion/modality_fusion.md)
- 视觉语言模型：对页面内容进行语义进行分析
- ali cosyvoice、f5-tts


### 训练数据利用
#### 集成学习
1. [Boosting](Ensemble/Ensemble/Boosting/boosting.md)

2. [Bagging](Ensemble/Ensemble/Bagging/bagging.md)
      - [Random Forests]
3. Stacking
4. Voting
5. [MoE](Ensemble/MoE/moe.md)
#### 数据增强
1. [dropout](Denoising/Dropout/dropout.md)
2. 加噪自编码器
    - [DAE](Denoising/DAE/dae.md)、[VAE](Denoising/VAE/vae.md)
    - [对抗训练](Denoising/AdversarialTraining/vat.md)
3. 数据增强  
    - flipping  
    - rotating  
    - transforming the color  
    - text_image + mask
#### 对比学习

### 大模型使用
#### [Pre-training]()
- scaling law失效 https://arxiv.org/pdf/2001.08361
- multitask learning
- batch train时，next token 为pad时，可以ignore此预测token的loss完全忽略PAD的影响
#### [SFT](LLM_Extend/LLM_SFT/LLM_SFT.md)
- prefix tuning
- Instrument Finetune
- Prompt Finetune
- RAdam
- hallucination幻觉

#### RLHF
#### [Distillation](LLM_Extend/distillation/distillation.md)
- ULMFit, Universal language model fine-tuning for text classification
- label smooth，标签平滑，可以进行cross-entropy，也可以使用$D_{KL}$
- switch transformer mixture hard and soft label: 0.25 of teacher and 0.75 of ground truth label



### 效率优化
#### 显存优化
- gradient checkpointing

#### Attention变种
- [MQA](Efficiency_Speedup/Attention_Variants/mqa.html#mqa)、[GQA](Efficiency_Speedup/Attention_Variants/mqa.html#gqa)、[MLA]
#### Attention效率优化    
- [flash attention](Efficiency_Speedup/Attention_Speedup/flash_attention.md)
- vLLM
- ollama
- KV cache https://blog.csdn.net/ningyanggege/article/details/134564203
#### 分布式训练模型
- [generate config]
- [DeepSpeed]: ZeRO: Zero Redundancy Optimizer
- [Megatron-LM]
- HAI-LLM framework（higher flyer）
#### [并行训练方案](Efficiency_Speedup/Parallelism/parallelism.md)
- TP(Tensor Parallelism)
- PP(Pipeline Parallelism)
- CP(Context Parallelism)
- DDP(Data Pparallelism)
  - DDP, Distributed Data Parallelism
  - FSDP, Fully Sharded Data Parallel，全切片数据并行
- MP(Model Parallelism)
- SP(Sequence Parallel)
#### [量化](Efficiency_Speedup/Quantization/quantization.md)
#### 模型推理优化
- onnx
- TensorRT

目标重要性衡量  

    - 比较去除目标的前后变化，一般差异值越大表示目标越重要  
    - AdaLoRA奇异值重要性衡量(equation 11)
