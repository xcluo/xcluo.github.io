---
title: "JavaScript"
---

- emit事件，用于触发监听事件，事件名作为第一个参数，事件参数作为第二个参数
- console.log结果输出在`F12 → 控制台 → 默认级别 → 信息`里

### 运行环境

- `Node.js`，JavaScript 运行时环境，基于 Chrome V8 引擎 
- `Next.js`：基于 Node.js的React 全栈框架

#### npm

默认自带包管理器 npm (node packages manager)，用于安装、分享和管理 JavaScript 包，包信息存储在packages.json中，具体包存放在node_modules文件夹中

- install 安装依赖包
```bash
pnpm install -g pnpm
```
- list 查看本地安装包
```bash
npm list -g --depth=0
```
- view 查看包信息
- update 更新包
- uninstall 卸载包
```
npm uninstall package_name
npm uninstall -g global_package_name
```
#### npx

npx (Node Package Executor) 为npm 5.2+ 自带的包执行工具

#### pnpm
pnpm (Performance npm) 兼容 npm 工作流、快速、磁盘空间高效的包管理器

`npm install -g pnpm`

### 语法相关

#### JavaScript 基础

- **变量声明**：~~`var`~~、`let`、`const`，不建议使用var声明变量

#### JavaScript 进阶

### 项目结构

- next.config.js，Next.js 配置
- tsconfig.json，TypeScript 配置


#### public

静态资源文件

#### scr

源代码

#### app

#### components

#### lib

工具函数和 Hook 结构

#### .next

Next.js 构建的动态输出结果

#### node_modules

依赖包安装路径

