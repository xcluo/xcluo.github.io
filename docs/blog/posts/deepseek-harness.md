---
date: 2026-09-17
slug: deepseek-harness
title: "DeepSeek Harness"
comments: true
categories:
  - Agent
tags:
  - Agent
  - DeepSeek
---
DeepSeek Harness 是一个强大的 AI 开发工具。

<!-- more -->

## 部署

```bash
git clone https://github.com/deepseek-ai/deepseek-harness.git
cd deepseek-harness
pnpm install
pnpm run build
pnpm dsh web
```

## 更新

```bash
git pull origin master
# 可以顺序执行，即 pnpm run clean && pnpm run build
pnpm run clean
pnpm run build
pnpm dsh web
```
