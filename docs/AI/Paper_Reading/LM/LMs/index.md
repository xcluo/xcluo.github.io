## 模型架构
### Encoder
- BERT
- [ELECTRA](Infrastructure/BERT/ELECTRA/electra.md)
    - 找错字
- [RoBERTa](Infrastructure/BERT/RoBERTa/roberta.md)
- MacBERT  
    - 该错字
    - 不再将token进行mask，而是对选定的token进行同义词、错别字、随机词替换，让模型进行纠正
    - 缓解预训练-微调差异： 消除了 [MASK] 令牌带来的不一致性
    - 针对中文进行了专门优化

- ALBERT
- DistillBERT
- BERT-wwm
- ViLBERT
### Causal/Prefix Decoder
- OpenAI: GPT
- Google: Gemini
- Anthropic: Cladue
- DeBERTa、DeBERTa_v3
- [LLaMA](Infrastructure/LLaMA/llama.md)
- [DeepSeek](Infrastructure/DeepSeek/deepseek.md)
- Mistral
- UniLM
- 阿里：通义千文
- 月之暗面Moonshot：Kimi
- 百川智能：百川大模型
- 百度：文心一言

- 智谱AI：GLM
- 昆仑万维：天工
- 腾讯：混元
- 华为：盘古
- 字节：豆包
- MiniMax：MiniMax

### Encoder-Decoder
- [MASS](Infrastructure/MASS/mass.md)
- [BART](Infrastructure/BART/bart.md)
- T5

## 下游任务
### [RAG](RAG/index.md)
- 向量增强
- 文本增强
- 问题
### [Agent](Agent/agent.md)



