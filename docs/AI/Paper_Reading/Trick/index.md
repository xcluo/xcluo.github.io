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
2. [Model Merging](Ensemble/Ensemble/Model_Merging/model_merging.md)
3. [Bagging](Ensemble/Ensemble/Bagging/bagging.md)
      - [Random Forests]
4. Stacking
5. Voting
6. [MoE](Ensemble/MoE/moe.md)
#### 数据增强
1. [Dropout](Denoising/dropout.md)
2. [Label Smoothing](Denoising/label_smoothing.md)
3. 加噪自编码器
    - [DAE](Denoising/DAE/dae.md)、[VAE](Denoising/VAE/vae.md)
    - [对抗训练](Denoising/AdversarialTraining/vat.md)
4. 数据增强  
    - flipping  
    - rotating  
    - transforming the color  
    - text_image + mask
5. [Subword Regularization](../Component/Tokenizer/SubWord/subword_tokenize.md#subword-regularization)、[BPE-Dropout](../Component/Tokenizer/SubWord/subword_tokenize.md#bpe-dropout)
#### 对比学习

### 大模型相关
#### Scaling Laws

#### [Pre-training]()
- scaling law失效 https://arxiv.org/pdf/2001.08361
- multitask learning
- batch train时，next token 为pad时，可以ignore此PAD token的loss以完全忽略PAD的影响
#### [SFT](LLM_Extend/LLM_SFT/LLM_SFT.md)
- prefix tuning
- Instrument Finetune
- Prompt Finetune
- RAdam
- hallucination幻觉

#### RLHF

#### PEFT
- ULMFit, Universal language model fine-tuning for text classification
- AdaLoRA奇异值重要性衡量(equation 11)

#### 蒸馏、压缩
- [软标签 & 硬标签](LLM_Extend/Distillation/distillation.md#soft-label-hard-label)
- [温度系数](LLM_Extend/Distillation/distillation.md#temperature)

### 模型压缩
- dim 压缩：[MRL](Efficiency_Speedup/Compression/mrl.md)
- 网络瘦身：[Slimmable Neural Networks](Efficiency_Speedup/Compression/slimmable_network.md)


#### Attention变种
- [MQA](Efficiency_Speedup/Attention_Variants/mqa.html#mqa)、[GQA](Efficiency_Speedup/Attention_Variants/mqa.html#gqa)、[MLA](../LM/LMs/Infrastructure/DeepSeek/deepseek.md#mla)
#### Attention效率优化
- [KV Cache](Efficiency_Speedup/Attention_Speedup/kv_cache.md)
- [flash attention](Efficiency_Speedup/Attention_Speedup/flash_attention.md)
- [vLLM](Efficiency_Speedup/Attention_Speedup/vllm.md)
- ollama

#### 显存优化
- [gradient checkpointing](Efficiency_Speedup/Quantization/gradient_checkpointing.md)
- Reducing activation recomputation in large transformer models

#### [并行训练](Efficiency_Speedup/Parallelism/parallelism.md)
- 数据并行 DP(Data Pparallelism)
    - DP，数据并行
    - DDP, Distributed Data Parallelism
    - FSDP, Fully Sharded Data Parallel，全切片数据并行
- 模型并行 MP(Model Parallelism)
    - TP(Tensor Parallelism)，如GEMM
    - PP(Pipeline Parallelism)
    - EP(Expert Parallelism)
- SP(Sequence Parallel)
- CP(Context Parallelism)
- 分布式训练
    - [generate config]
    - [DeepSpeed](../../AI_Platform/Microsoft/deepspeed.md): ZeRO: Zero Redundancy Optimizer
    - [Megatron-LM](../../AI_Platform/Nvidia/megatron-lm.md)
    - HAI-LLM framework（higher flyer）


#### 推理优化
- [量化](Efficiency_Speedup/Quantization/quantization.md)
- onnx：open neural network exchange
- TensorRT：

目标重要性衡量  

    - 比较去除目标的前后变化，一般差异值越大表示目标越重要  
