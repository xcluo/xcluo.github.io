---
title: "npm"
---


### npm

默认自带包管理器 npm (node packages manager)，用于安装、分享和管理 JavaScript 包，包信息存储在packages.json中，具体包存放在node_modules文件夹中

#### install 安装包

基本语法 `npm install [OPTIONS] [PACKAGE]`

```bash
# 本地安装（默认）
npm install <package_name>
npm install <package_name>@<version>        # 指定版本
npm install <package_name>@latest           # 最新版本
npm install <package_name>@">=1.0.0 <2.0.0" # 版本范围

# 全局安装
npm install <package_name> -g

# 从不同来源安装
npm install <package_name> --registry=https://registry.npmmirror.com  # 镜像源
npm install <package_name> --git https://github.com/user/repo  # 从 Git 安装
npm install <package_name> --git-tag  # 按 tag 安装

# 其他常用选项
npm install                      # 安装 package.json 中所有依赖
npm install --force              # 强制重新安装
npm install --legacy-peer-deps   # 忽略 peer dependency 冲突
npm install --ignore-scripts     # 忽略安装脚本
npm install --dry-run            # 模拟安装（不实际安装）
```

#### list 查看本地安装包

```bash
npm list                          # 查看本地依赖树
npm list -g --depth=0             # 查看全局安装的包（不显示依赖）
npm list <package_name>           # 查看特定包信息
npm list --depth=1                # 查看依赖树深度
npm list --prod                   # 只显示 dependencies
npm list --dev                    # 只显示 devDependencies
```

#### view 查看包信息

```bash
npm view <package_name>           # 查看包信息
npm view <package_name> version   # 查看最新版本
npm view <package_name> versions  # 查看所有版本
npm view <package_name> repository.url  # 查看仓库地址
npm view <package_name> homepage  # 查看主页
```

#### update 更新包

```bash
npm update                        # 更新所有包
npm update <package_name>         # 更新特定包
npm update -g                     # 更新全局包
npm outdated                      # 查看可更新的包
```

#### uninstall 卸载包

```bash
# 本地卸载
npm uninstall <package_name>              # 卸载并从 package.json 移除
npm uninstall <package_name> --save-optional  # 从 optionalDependencies 移除

# 全局卸载
npm uninstall -g <package_name>

# 其他选项
npm uninstall --force             # 强制卸载
npm uninstall --legacy-peer-deps  # 忽略 peer dependency 冲突
```
> 删除时，只需要要填写package_name，无需指定版本号

#### run 执行脚本

```bash
# 执行 package.json 中 scripts部分定义的脚本
npm run <script_name>

# 常用内置脚本，对应.scripts部分的所有key（可省略 run）
npm run build              # 构建
npm run dev / npm run dev  # 开发
npm run start              # 启动
npm run test               # 测试
npm run clean              # 常用来清理构建产物、缓存或临时文件

# 常用选项
npm run --silent           # 静默模式，不输出
npm run --if-present       # 脚本不存在时不报错
npm run --ignore-scripts   # 忽略脚本中的子脚本
npm run -- <args>          # 传递参数给脚本

# 示例：传递参数
npm run dev -- --port 3000
npm run build -- --env production

# 查看所有可用脚本
npm run
npm run --list             # 列出所有脚本
```

```json "scripts 示例"
{
  "scripts": {
    "dev": "vite",
    "build": "vite build",
    "preview": "vite preview",
    "test": "vitest",
    "lint": "eslint .",
    "clean": "rm -rf dist node_modules",
    "postinstall": "husky install",
    "prebuild": "node scripts/check.js",
    "build:prod": "NODE_ENV=production vite build"
  }
}
```

##### 多命令执行

```bash
# 顺序执行
npm run clean && npm run build
npm-run-all --sequential clean build  # 顺序

# 并行执行
npm run build:css & npm run build:js
npm run clean & npm run build

# 使用 npm-run-all（第三方库）
npm install --save-dev npm-run-all
```

### npx

npx (Node Package Executor) 为npm 5.2+ 自带的包执行工具。“一次性命令运行器”：本地有就用本地，本地没有就临时下载，执行完不留下项目依赖。适合偶尔用一次的命令行工具，比如脚手架、格式化工具、代码生成器等。

### pnpm

pnpm (Performance npm) 兼容 npm 工作流（通用package.json），快速、磁盘空间高效的包管理器

!!! info
    - npm会对每一个项目下载package.json中的包，每个项目独立，且会重新下载
    - pnpm会对所有pnpm项目构建一个共享缓存库，当package.json的包在缓存库中时，会自动索引；否则下载至共享缓存库中再索引

- `npm install -g svg-to-ico`，svg-to-ico input-file-name [output-file-name]