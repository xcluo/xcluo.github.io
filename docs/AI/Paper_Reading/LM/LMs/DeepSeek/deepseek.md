### DeepSeek-v2
> 论文：DeepSeek-V2: A Strong, Economical, and Efficient Mixture-of-Experts Language Model  
> DeepSeek-AI, 2024


训练：

1. full pre-trained on 8.1T tokens(DeepSeek 67B corpus + Chinese Data + higher quality data)  
2. 1.5M conventional sessions with various domains such math, code, writing, reasoning, safety, and more to SFT DeepSeek-v2 chat  
3. follow DeepSeekMath to employ Group Relative policy Optimization(GRPO) to align model with RLHF


模型架构：

1. MLA：Multi-head Latent Attention  
2. DeepSeekMoE
3. DeepSeek-V2
4. DeepSeek-V2-Lite
5. DeepSeek-V2-Chat_SFT
6. DeepSeek-V2-Chat_RL

策略：
1. Load Balance Loss
   - expert-level balance loss: 类似于$L_{importance}$
   - Device(gpus or tpus)-Level Balance Loss: 
   - Communication Balance Loss
2. Token-Dropping Strategy: In this way, we can flexibly decide whether to drop tokens during inference according to the efficiency requirements, and always ensure consistency between training and inference.  
3. R1中的reward model和v2中的不相同，实际上是一个rulee-based system

3. HAI-LLM framework

数据处理：
1. Data Construction
2. BBPE（Byte-level Byte-Pair Encoding）

- MTP: 类似于skip-gram，t预测t+1, t+2, ..., t+k

- low-precision training