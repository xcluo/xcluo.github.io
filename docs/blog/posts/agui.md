---
draft: true 
date: 
  created: 2026-01-30
slug: agui
title: "AG-UI"
comments: true
categories:
  - AI应用
---
[AG-UI](https://github.com/ag-ui-protocol/ag-ui)

<!-- more -->

## 快速启动
可参照支持的框架对应的文档docs进行启动部署

1. git clone项目
2. 创建项目路径 `npx copilotkit@latest create -f langgraph-py`，（需指定项目名）
3. 进入项目路径，安装依赖包 `npm install`
4. **进入agent路径**，修改`main.py`中大模型配置，`llm=ChatOpenAI(...)` 从而指定智能体使用的大模型
5. 同步启动前、后端 `npm run dev`