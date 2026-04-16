---
title: "skill"
---

- [《openclaw技能使用分享》](https://www.yuque.com/ruishi-7yym8/mqxshg/bnaufv0n0wd6oa5h?singleDoc#)

## 架构介绍

```bash
skill-name/
├── SKILL.md                    # 必需：技能定义和指令
│   ├── YAML frontmatter        # name, description（触发关键）
│   └── Markdown body           # 详细工作流说明
│
└── Bundled Resources/          # 可选：辅助资源
    ├── scripts/                # 可执行脚本（Python/Bash等）
    │   └── helper.py
    ├── references/             # 参考文档（按需加载）
    │   ├── aws.md
    │   └── gcp.md
    └── assets/                 # 模板、图标、字体等
        └── template.docx
```

### SKILL.md

#### YAML Metadata

yaml格式的文档说明

- `name` 技能名称
- `description` 技能描述===（最为重要，模型选择该技能前，提示词只加载这一部分）===，当用户输入的自然语言符合该技能功能描述时，会触发该技能调用。
- `compatibility` 可选，依赖要求

#### Markdown Body

markdown格式的文档说明，详细描述技能的工作流、参数、输入示例、返回值等。

### Scripts

### references

存放需要被 AI 阅读理解的参考文档，通常按领域/场景拆分。即当AI选用该技能时，针对特定场景，在SKILL.md中会提示参考具体文档内容。

### assets

存放不需要 AI 阅读、直接作为输出或展示使用的静态文件。
