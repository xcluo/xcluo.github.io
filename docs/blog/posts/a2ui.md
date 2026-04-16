---
draft: true 
date: 
  created: 2026-01-30
slug: a2ui  # url别名
title: "A2UI"
comments: true
categories:
  - AI应用
---
[A2UI](https://github.com/google/A2UI) 按照用户的需求，给出一些可选表单让用户一次性填完关键信息、指定参数，而不用大模型多轮乒乓交互，盲人摸象般地高负载串行

<!-- more -->

## 快速启动

1. git clone项目并进入仓库

后端相关

- 进入点餐路径`cd samples/agent/adk/restaurant_finder`
- 安装python环境`uv venv`，`uv sync`
- 将`agent.py → _build_agent`中的LlmAgent初始化第一个行改为 `model=LiteLlm(model="openai/{MODEL_NAME}", api_base="{API_BASE_V1}", api_key="{API_KEY}")`以自定义大模型使用
- 启动后端服务 `uv run .`

前端相关

- 进入路径 `cd renderers/web_core`
- 安装依赖包并构建 `npm install`，`npm run build`
- 进入路径 `cd renderers/lit`
- 安装依赖包并构建 `npm install`，`npm run build`
- 进入路径 `cd samples/client/lit/shell`
- 安装依赖 `npm install`
- 启动前端服务 `npm run dev`