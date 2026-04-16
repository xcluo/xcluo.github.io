---
date: 
  created: 2024-01-31
  updated: 2024-02-28
title: "CoPaw" 
---
<!-- more -->

### 部署

#### 安装

1. uv创建并激活虚拟环境
2. 安装`pip`以及`copaw`

#### 初始化

使用`copaw init --defaults`命令生成工作目录`~/.copaw`，包括

- `config.json` 基础配置信息，如channels配置
- `HEARTBEAT.md` 心跳检查清单，定义定期自检任务，用于系统健康监控和状态报告。较cron定时技能更注重多个检查任务合并为一个任务，且对时间精度要求没那么高，即更倾向将 cron 用于精确调度和独立任务
- `chats.json` 存储历史对话列表信息，即前端左侧历史对话记录
- `MEMORY.md` 长期记忆存储，保存重要的决策、偏好和极少变动的关键信息，通过 `write/edit` 文件工具由 Agent 自动维护，支持语义搜索，便于快速检索历史信息
> 当用户明确说"记住这个"时立即写入时会进行更新
- `memory/` 按日期存储每日的工作记录和运行上下文，`memory/YYYY-MM-DD.md`
> 当对话上下文 token 超过阈值时，系统自动将对话摘要写入当日日志
- `SOUL.md` （Agent特点）定义 AI 助手的性格、说话风格和人格特质，如设定回复的语气和风格、定义行为准则和价值观等
- `PROFILE.md` 存储Agent特点（名称、角色、风格）以及用户信息（用户姓名、时区，个人偏好和背景）
- `AGENTS.md` 系统提示词主体之一，定义 Agent 的核心能力、列出可用的工具和技能，在每次会话开始时重新加载，确保使用最新配置。

### 运用

#### 运行服务

启动copaw服务，本质为uvicorn run ASGI服务器

```bash
copaw app \
--host 127.0.0.1 \
--port 8088
```

#### 配置provider

配置文件路径 `copaw(python包安装路径) → providers → providers.json`
> 网页交互页面配置更为便捷，`Model ID`与`Model Name`不能相同，否则无法加载成功

```json
"custom_providers": {
    "zai": {
      "id": "zai",
      "name": "zai",
      "default_base_url": "https://open.bigmodel.cn/api/paas/v4",
      "api_key_prefix": "",
      "models": [
        {
          "id": "glm-4.7-flash",
          "name": "glm-4.7-flashglm-4.7-flash"
        }
      ],
      "base_url": "https://open.bigmodel.cn/api/paas/v4",
      "api_key": "you_api_key",
      "chat_model": "OpenAIChatModel"
    }
  },
  "active_llm": {
    "provider_id": "zai",
    "model": "glm-4.7-flash"
  }
```

### 智能体处理流程

#### ENV Context

1. session_id, user_id, channel及工作目录信息
2. 提示完成任务优先考虑skills
3. 提示通过读取skill文档了解skill功能
4. 补充当前MCP信息

#### System Prompt

若 `.copaw/config.json` 存在键`system_prompt_files`，则加载该键对应的值，作为system prompt；否则以文件`AGENTS.md, SOUL.md, PROFILE.md` 为主体作为system prompt

#### ReAct过程

- 当对话token数超过一定阈值时，触发压缩，每次与AI会话前会尝试保留最近k次对话记录作为短期记忆直接原文作为prompt，其余记录执行压缩为长期记忆作为prompt
- ReAct Agent，循环迭代，根据当前情况，考虑下一步应该做什么，知道超出最大迭代次数或任务完成
