- llama 1-2, A100, 1.8 T tokens for llama 2
- llama 3, 15T tokens, 16k H100 (405B)
    - dense transformer rather than moe transformer（trained and not fine-tuned）
    - 大力出奇迹且简单：sft + RS(reject sampling) + DPO，效果提升主要还是数据质量高和多样性
    - 70b is comparable better
- TP(tensor parallelism), PP(pipeline parallelism), CP(context parallelism), DP(data parallelism)
- Guard 3可以多prompt的输入输出进行一些安全上的改写

preference data

- 使用多个模型对给定prompt进行生成，并采样两条样本（由不同模型生成）