#### 多模态融合 & 多任务学习
- [token](Multimodality_Fusion/token_modality.md)
- [sound](Multimodality_Fusion/sound_modality.md)
- [shape](Multimodality_Fusion/shape_modality.md)
- [fusion](Multimodality_Fusion/modality_fusion.md)
- 视觉语言模型：对页面内容进行语义进行分析
- hallucination幻觉
- ali cosyvoice、f5-tts

####  Attention Variants & Proxy Task
- multitask learning

#### 集成学习 & 数据增强 & 对比学习
- [Boosting](Ensemble/Ensemble/Boosting/boosting.md)

- [Bagging](Ensemble/Ensemble/Bagging/bagging.md)
      - [Random Forests]
- Stacking
- Voting
- [MoE](Ensemble/MoE/moe.md)
- [dropout](Denoising/Dropout/dropout.md)
- [DAE](Denoising/DAE/dae.md)
- [VAE](Denoising/VAE/vae.md)
- [Adversarial Training](Denoising/AdversarialTraining/vat.md)
- 数据增强  
    - flipping  
    - rotating  
    - transforming the color  
    - text_image + mask
  - 对比学习

#### 大模型使用
- [蒸馏](LLM_Extend/distillation/distillation.md)
  - label smooth，标签平滑
  - switch transformer mixture hard and soft label: 0.25 of teacher and 0.75 of ground truth label
  - ULMFit, Universal language model fine-tuning for text classification
- [LLM_SFT](LLM_Extend/LLM_SFT/LLM_SFT.md)
- prefix tuning
- Instrument Finetune
- Prompt Finetune
- gradient clip
- RAdam
- scaling law失效 https://arxiv.org/pdf/2001.08361
- https://arxiv.org/pdf/2409.14781 Pretraining Data Detection for Large Language Models: A Divergence-based Calibration Method
- https://arxiv.org/pdf/2404.02655 Calibrating the Confidence of Large Language Models by Eliciting Fidelity
- deepseek

#### 存储优化
- Attention优化
    - [flash attention](Memory_Saving/Flash_Attention/FlashAttention.md)
    - [MQA](Memory_Saving/Attention_Variants/mqa/#mqa)、[GQA](Memory_Saving/Attention_Variants/mga/#gqa)
    - KV cache https://blog.csdn.net/ningyanggege/article/details/134564203
- 分布式训练
    - [DeepSpeed]
    - [Megatron-LM]
- [Parallelism](Memory_Saving/Parallelism/parallelism.md)
    - TP(Tensor Parallelism)
    - PP(Pipeline Parallelism)
    - CP(Context Parallelism)
    - DP(Data Pparallelism)
    - MP(Model Parallelism)
    - SP(Sequence Parallel)
- 量化 [Quantization](Memory_Saving/Quantization/quantization.md)


目标重要性衡量  

    - 比较去除目标的前后变化，一般差异值越大表示目标越重要  
    - AdaLoRA奇异值重要性衡量(equation 11)
