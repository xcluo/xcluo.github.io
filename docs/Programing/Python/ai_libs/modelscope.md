---
title: "modelscope"
---

# ModelScope

阿里云的模型平台，提供模型下载、模型推理、微调训练等功能。

## 安装

```bash
pip install modelscope
```

## 基础用法


### 命令行下载

```bash
modelscope download --model Qwen/Qwen2.5-7B-Instruct    # 下载modelscope中指定的模型
```
> 默认存储在 `~/.cache/modelscope/hub/models/`
