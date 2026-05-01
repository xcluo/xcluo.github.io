### 深度学习框架

- [ ] SafeTensors

#### Tensorflow

- [Tensorflow](Tensorflow/index.md)

#### Pytorch

- [PyTorch](PyTorch/index.md)
- [Transformers](Transformers/transformers.md)

#### PaddlePaddle

- [PaddlePaddle](PaddlePaddle/index.md)
- [Nvidia](Nvidia/index.md)

### 数据预处理

1. `tf1_dataset_utils.py`
2. `torch_dataset_utils.py`

3. dataset generator
4. Transformers + datasets

### scheduler

### 分布式训练框架

- [DeepSpeed](DeepSpeed/deepspeed.md)
- [Megatron-LM](Nvidia/megatron-lm.md)
- HAI-LLM

### 运行部署框架

#### 部署

- [vLLM](../Paper_Reading/Trick/Efficiency_Speedup/Attention_Speedup/vllm.md)
- [TensorRT](Nvidia/tensorrt.md)
- triton

#### +开发

- [Ollama](Ollama/ollama.md)
- [LangChain](LangChain/langchain.md)
- [Dify](Dify/dify.md)
- [RAGFlow](RAGFlow/ragflow.md)
- Haystack
- 阿里百炼

### 大模型调用

#### text_llm

```python
import requests

API_KEY = your_api_key
BASE_URL = api_url              # include "/chat/completions"
MODEL_NAME = llm_model_name

res = requests.post(
    BASE_URL,
    json={
        "model": MODEL_NAME,
        "messages" : [
            {"role": "user", "content": "你是哪个模型？"}
        ],
        # "max_tokens": 512,
        "stream": True,
        "temperature": 0.6,
        "top_p": 0.9,
        "enable_thinking": True
    },
    headers={
        "Content-Type": "application/json",
        "Authorization": f"Bearer {API_KEY}"
    }
)
```

#### multi-modality_llm

=== "RESTful API"
    ```python
    import requests
    import base64
    import json

    API_KEY = your_api_key
    BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions"              # include "/chat/completions"
    MODEL_NAME = "qwen3-omni-flash"

    pcm_data = open(pcm_file_path, "rb").read()
    audio_b64 = base64.b64encode(pcm_data).decode('utf-8')
    image_file = open(image_path, "rb")
    image_b64 = base64.b64encode(image_file.read()).decode('utf-8')

    res = requests.post(
        BASE_URL,
        json={
            "model": MODEL_NAME,
            "messages": [
                {
                    "role": "system",
                    "content": [
                        {
                            "type": "text",
                            "text": system_prompt
                        }
                    ]
                },
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "input_audio",
                            "input_audio": {
                                "data": f"data:audio/pcm;base64,{audio_b64}",
                                "format": "pcm"
                            }
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{image_b64}"
                            }
                        }
                    ]
                }
            ],
            "stream": True,             # 强制为True
            "temperature": 0.6,
            "top_p": 0.9,
            "enable_thinking": True
        },
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {API_KEY}"
        }
    )

    prefix_span = "data: "              # 非空chunk均使用该前缀
    reasoning_content = ""
    content = ""
    for p in res.iter_lines():
        p = p.decode("utf-8")
        if not p.strip() or p[len(prefix_span):].strip() == "[DONE]":
            continue
        data = json.loads(p[len(prefix_span):])
        if "reasoning_content" in data["choices"][0]["delta"] and data["choices"][0]["delta"]["reasoning_content"]:
            reasoning_content += data["choices"][0]["delta"]["reasoning_content"]
        if "content" in data["choices"][0]["delta"] and data["choices"][0]["delta"]["content"]:
            content += data["choices"][0]["delta"]["content"]
    ```

#### Agent

#### RAG
