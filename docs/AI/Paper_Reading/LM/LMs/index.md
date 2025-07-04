## 模型架构
### Encoder
- BERT
- [ELECTRA](Infrastructure/BERT/ELECTRA/electra.md)
- [RoBERTa](Infrastructure/BERT/RoBERTa/roberta.md)

### Decoder
- OpenAI: GPT
- Google: Gemini
- thropic: Cladue
- DeBERTa、DeBERTa_v3
- [LLaMA](Infrastructure/LLaMA/llama.md)
- [DeepSeek](Infrastructure/DeepSeek/deepseek.md)
- Mistral
- Manus
- 阿里：通义千文
- 月之暗面Moonshot：Kimi
- 百川智能：百川大模型
- 百度：文心一言



- 智谱AI：GLM

    ```python title="翻译"
    from zhipuai import ZhipuAI
    import json


    api_key = "my_api_key"
    client = ZhipuAI(api_key=api_key)

    def get_response(sample_input):
        response = client.chat.completions.create(
            model="GLM-4-Flash",  # 请填写您要调用的模型名称
            messages=[
                {"role": "user", "content": "你是一个翻译专家，你需要将用户输入的json格式中content对应的文本翻译为中文，并将中文翻译结果作为该json样本中\"t\"键对应的值，结果返回json格式"},
                {"role": "user", "content": json.dumps(sample, ensure_ascii=False)},
            ],
        )

        return response.choices[0].message.content

    response = get_response(line)
    # 返回格式为 ```json ... ```
    left_index, right_index = response.find("{"), response.rfind("}")
    trans_line = json.loads(response[left_index: right_index + 1])
    if trans_line["content"] != line["content"]:
        continue
    ```

- 讯飞：星火
- 昆仑万维：天工
- 腾讯：混元
- 华为：盘古
- 字节：豆包
- MiniMax：MiniMax

### Prefix LM
- UniLM

### Encoder-Decoder
- [MASS](Infrastructure/MASS/mass.md)
- [BART](Infrastructure/BART/bart.md)
- T5

## 下游任务
### [RAG](RAG/index.md)




