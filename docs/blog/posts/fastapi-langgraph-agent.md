---
date: 2026-09-29
slug: fastapi-langgraph-agent
title: "FastAPI + LangGraph"
comments: true
categories:
  - AI
tags:
  - Agent
  - LangGraph
  - FastAPI
---
FastAPI 与 LangGraph 的结合使用。

<!-- more -->

## 简介

LangGraph 是一个用于构建有状态、多参与者应用的框架，基于 LangChain 构建。

## 安装

```bash
# 轻量版https://github.com/Dopetaiga/fastapi-langgraph-agent
# 完全版https://github.com/wassim249/fastapi-langgraph-agent-production-ready-template
git clone https://github.com/wassim249/fastapi-langgraph-agent-production-ready-template.git
cp .env.example .env.devolpment   # 填写OPENAI_API_KEY、OPENAI_BASE_URL以及DEFAULT_LLM_MODEL
make install                      # 也可根据Makefile进行手动操作
make docker-up                    # docker部署postgresql 和 app/服务，可也根据Makefile进行手动操作

# 本地部署app/服务
source ./scripts/set_env.sh ${ENV}    # ENV=development
export PYTHONUTF8=1
修改.env.${ENV}中的 `POSTGRES_HOST=localhost`
uv run uvicorn app.main:app --reload --port 8000
```
