## GPT_v1

## GPT_V2

## InstructGPT
> 论文：Training language models to follow instructions with human feedback  
> OpenAI, 2022 Mar, NeurIPS 2022

### 主要内容
#### Model Details
1. SFT
2. Reward Model Training: train on reward model training set

    - 对lr和schedule不敏感：lr降低50%的结果也相似
    - 对#epoch敏感，大epoch容易过拟合
    - 奖励模型pair-rank-loss，每个prompt有K个completions，每次通过两两对比学习来进行比较训练，因此有$C(K, 2)$个组合数
    - bs = M * C(K, 2)

3. initialization models for RLHF
4. RLHF Training


#### Prompt Data Details
1. Labeler-written prompts，we asked contractors to write below three kinds of prompts:  
    1. Plain: 
    2. Few-shot: 
    3. User-based: 

2. API used prompts
3. Dataset sizes
4. Data diversity
#### Human Data Collection Details

## GPT_v3

