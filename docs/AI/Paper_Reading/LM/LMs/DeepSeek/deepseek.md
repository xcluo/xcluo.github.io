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