---
title: Docker
---


#### 安装docker

- 更新wsl系统：`wsl --update --web-download` 
- 安装Docker：`"Docker Desktop Installer.exe" install --installation-dir=E:\Docker\Desktop --wsl-default-data-root=E:\Docker\DockerDesktopWSL --windows-containers-default-data-root=E:\Docker\Containers`  
    
    - 指定安装路径：`--installation-dir=install_absolute_dir`
    - 指定镜像存储路径（需与安装路径不同）：`--wsl-default-data-root=wsl_absolute_dir\DockerDesktopWSL`
    - `--windows-containers-default-data-root=containers_absolute_dir`
    
    > 如果无法弹出安装页面,可能是历史安装时使该文件夹需要管理员权限才能写入,可通过删除文件夹后重新启动安装 

- 配置Docker引擎加速器(可通过切换源的顺序完成pull)  
    ```
    "registry-mirrors": [
        "https://docker.xuanyuan.me"
    ]
    ```

    1. [`登录SWR`](https://support.huaweicloud.com/usermanual-swr/swr_01_0022.html) `
    2. 选择区域局点：华东-上海 
    3. 左侧镜像资源 → 镜像中心 → 右上角镜像加速器 →加速器地址
    4. 复制至 `"registry-mirrors": [url]`  

- 汉化
    1. [`汉化包release地址`](https://github.com/asxez/DockerDesktop-CN/releases)   
    2. 选择指定版本  
    3. 替换`~/frontend/resources/app.asar`

### 信息查看

#### `version`

`docker version` 查看docker版本及desktop版本

#### `info`

`docker info` 查看docker详细信息

#### `inspect`

`docker inspect` 本地查询镜像/容器信息

#### Dockerfile

用于定义镜像构建流程的文本配置文件，通过 [`docker build`](#build) 命令可读取 Dockerfile 并自动构建自定义镜像。

高频指定关键字（出于规范和可读性全大写）

- `FROM image_name:tag` 基础镜像，可通过多行From命令包含多个基础镜像
- `WORKDIR image_dir` 指定镜像工作目录，即`RUN`、`COPY`、`CMD`、`ENTRYPOINT` 等指令默认工作目录
- `RUN apt update && apt install -y net-tools` 在镜像构建阶段执行指定命令
- `COPY host_file image_file` 拷贝主机文件到镜像，默认文件拥有者是root，可通过 `--chown` 更改文件拥有者

    > `ADD` 为增强版兼容 `COPY` 的命令

- `ENV VAR_NAME=VAR_VALUE` 设置环境变量
- `ARG VAR_NAME` 定义变量
- `USER user_name` 切换用户（一般使用 `RUN useradd -m user_name` 先创建用户）
- `EXPOSE port_num` 声明容器运行时对外暴露的网络端口（仅用于文档说明和镜像使用者参考，不做实际映射）
- `ENTRYPOINT` 指定容器入口命令

    > `ENTRYPOINT` 在 `CMD` 前先执行，且不容易被覆盖

- `CMD` 容器启动执行的命令（一个 Dockerfile 中仅能有一个有效 CMD，多个则对最后一个生效）

#### .dockerignore

存放在 Dockerfile 同目录下，用于构建 Docker 镜像时排除配置文件，类似于 .gitignore

#### docker-compose.y(a)ml

YAML格式，依赖缩进（2个空格而不是TAB），大小写敏感，#为注释

- `versions` 指定Compose文件格式版本
- `services` 定义容器
    - `<service_name>` 服务名
        - `image` 指定初始化容器的镜像
        - `build` 容器构建方法
            - `context` 指定构建上下文路径  （可以是远程 Git 仓库）
            - `dockerfile` 参考的容器构建文件
            - `args` 传递给dockfile的ARG
            - `network` 构建时使用的网络模式
        - `container_name` 指定容器名
        - `ports` 指定端口映射
        - `volumes` 挂载数据卷
        - `networks` 指定容器所属网络
        - `depends_on` 依赖关系，先启动依赖的服务后再启动当前服务
        - `restart` 容器重启策略
        - `healthcheck` 
        - `environment` 设定环境变量
- `networks` 定义网络
- `volumes` 定义外部数据卷（如果使用本地路径，则无需在该部分声明）
    ```
    volumes:
        volume_name_1:
            external: true      # 已有数据卷
        volume_name_2:          # 默认为false，表示新建数据卷
    ```

### Image相关命令

#### `images`

列出镜像信息，基本语法 `docker images [OPTIONS] [REPOSITORY[:TAG]]`

Options

- `-a` -all，包括中间层的所有镜像  
- `-f cond` --filter，执行信息过滤，常用条件为`status=running`， `name=my-nginx` 等
- `-q` --quiet，只显示镜像 ID

Repository

- 镜像仓库/镜像名

#### `search`

基本语法 `docker search [OPTIONS] TERM`，从Docker Hub查找公开镜像

Options

- `-f=stars=1000` --filter 按条件过滤搜索结果，支持以下过滤条件  
    - stars=<number> 星标数
    - is-official=<true/false> 是否为官方镜像
    - is-automated=<true/false> 是否为自动化构建
    - `--filter=cond1 --filter=cond2` 多条件过滤
- `--limit=25` 限制返回结果数量（默认25，最大100）
- `--no-trunc` 显示完整的描述信息（不截断）
- `--format "table {{.Name}}\t{{.Description}}"` 自定义输出格式（Go模板）

#### `pull`

未指定标签，默认拉取lastest

#### `save`

导出镜像，基本语法 `docker save [OPTIONS] IMAGE [IMAGE...]`

Options

- `-o file.tar` --output 输出至镜像存档文件  

    > 等价于 `docker save IMAGE > file.tar`

Image：可为 `image_name:tag`（推荐）、`image_id`

#### `load`

加载导入镜像  `docker load [OPTIONS]`

Options

- `-i file.tar` --input 输入镜像存档文件  
- `-q` --quiet，忽略加载输出

#### `build`

从 Dockerfile （首字母大写，无后缀）文件中自定义构建镜像，基本语法 `docker build [OPTIONS] PATH|URL|-`

Options  

- `--build-arg NEXT_PUBLIC_API_URL=YOUR_LANGMANUS_API` 传递构建时环境变量参数
- `t image_name:tag` --tag 给镜像命名打标签，未指定时默认为`latest`
- `--no-cache` 不适用缓存，强制重新构建
- `-f file_name` --file 指定Dockerfile路径（未指定则查找当前目录）

#### `tag`

为镜像添加/修改名字，基本语法 `docker tag SOURCE_IMAGE[:TAG] TARGET_IMAGE[:TAG]`

#### `rmi`

删除镜像 `docker rmi [OPTIONS] IMAGE [IMAGE...]`
> `docker image prune` 自动筛选删除悬空无用镜像

Options

- `-f` --force，强制删除

Image：可为 `image_name:tag`、`image_id`，若为image_name则默认删除`image_name:latest`

### Container相关命令

#### `ps`

列出本地 Docker 环境中（默认为运行中）的容器信息，基本语法 `docker ps [OPTIONS]`

Options

- `-a` -all，包括未运行的所有容器
- `-f cond` --filter，执行信息过滤，常用条件为状态`status=running`， 容器名`name=my-nginx`, 端口号 `publish=8080` 等
- `-q` --quiet，只显示容器 ID
- `-n N` --last，显示最新创建的N个容器

#### `stats`

（每秒刷新一次）实时监控容器资源使用情况，基本语法 `docker stats [OPTIONS] [CONTAINER...]`

Options

- `-a` --all，显示所有容器
- `--no-stream` 输出一次数据后立即退出

#### `create`

基于镜像创建容器，基本语法 `docker create [OPTIONS] IMAGE [COMMAND] [ARG...]`

Options

- `--name container_name` 给容器命名
- `-e DB_HOST=db` --env，设置环境变量
- `-p host_port:container_port` --publish，宿主机和容器端口映射
- `-v host_dir:container_dir[:挂载权限]` --volume，挂载卷，将宿主机目录和容器目录绑定（宿主机和容器数据共享），用于持续性保持容器数据，==`host_dir` 必须为绝对路径或以`./`开头的相对路径==
    > `host_dir` 不为宿主机路径时，会自动创建一个同名卷柜
    > 挂载权限 `{ro: 只读, ++rw++: 读写}`

```bash
# 挂载docker命令 + 套接字
-v /usr/bin/docker:/usr/bin/docker \
-v /var/run/docker.sock:/var/run/docker.sock
```

#### `run`

启动容器（若无容器先执行create），基本语法 `docker run [OPTIONS] IMAGE [COMMAND] [ARG...]`

Options  

- `-d` 后台运行
- `-it` 交互模式（带终端）
- `-d`, --detach 后台运行
- `--rm` 退出运行后自动删除容器及其关联的匿名卷（命名卷不删除）
- `--restart strategy` 重启策略，`{++no++: 不重启, on-failure[:N]: 仅异常时重启（最大重启次数为N，未指定时无限制）, unless-stopped: 仅手动停止后不重启, always: 随docker服务持续运行}`
- `--name container_name` 给容器命名
- `-p host_port:container_port` --publish，宿主机和容器端口映射
量
- `-e DB_HOST=db` --env，设置环境变量

    > 公开镜像中，文档内一般有可设定环境变量参数介绍

- `--env-file .env` 从文件读取环境变
- `-v host_dir:container_dir[:挂载权限]` --volume，挂载卷，将宿主机目录和容器目录绑定（宿主机和容器数据共享），用于持续性保持容器数据

    > 挂载权限 `{ro: 只读, ++rw++: 读写}`

#### `volume`

管理数据卷（匿名卷 / 命名卷），基本语法 `docker volume COMMAND`  
> 命名卷：指定了volume_name的卷；匿名卷：未指定volume_name直接挂载的卷

Command

- `create volume_name` 创建命名卷
- `inspect volume_name|sha_id [volume_name|sha_id]` 查看数据卷详情
    - `docker inspect container_id | jq -c .[0] | jq -c .Mounts` 获取容器挂载卷信息
- `ls` 列出所有数据卷
- `prune` 批量清理所有未被容器关联的数据卷
- `rm volume_name|sha_id [volume_name|sha_id]` 删除数据卷

#### `start`

启动容器，基本语法 `docker start [OPTIONS] CONTAINER [CONTAINER...]`

Options

- `-i` --interactivate，与容器终端交互模式启动

#### `stop`

停止容器运行，基本语法 `docker stop [OPTIONS] CONTAINER [CONTAINER...]`

Options

- `-t senc` --timeout，等待senc秒后停止容器，可防止立即停止数据异常

#### `kill`

强制停止容器运行，基本语法 `docker stop [OPTIONS] CONTAINER [CONTAINER...]`

#### `restart`

快速重启运行中（或已停止）容器，基本语法 `docker restart [OPTIONS] CONTAINER [CONTAINER...]`

Options

- `-t senc` --timeout，等待senc秒后重启容器，可防止立即重启数据异常

#### `exec`

在运行中的容器内执行指定命令，基本语法 `docker exec [OPTIONS] CONTAINER COMMAND [ARG...]`

Options

- `-i` --interactive，保持容器的标准输入（STDIN）打开，允许用户向容器内命令传递输入内容
- `-t` --tty，为容器内的命令分配一个伪终端（TTY），模拟真实终端环境
- `-d` --detach，后台执行命令
- `-u <name|uid>[:<group|gid>]` --user，以指定用户身份执行命令
- `-w container_dir` --workdir，指定命令工作目录

Command（常为linux常用命令）

- `/bin/bash` 打开容器bash终端
- `/bin/sh` 打开容器sh终端
 
#### `logs`

查看容器日志输出，基本语法 `docker logs [OPTIONS] CONTAINER`

Options

- `-f` --follow，实时输出容器日志，类似于`tail -f`
- `-n N` --tail，输出最后N条日志，类似于`tail -N` 
- `--since timestamp` 开始时间，相对时间时往前倒
- `--until timestamp` 截至时间，相对时间时往后倒

    > 绝对时间：YYYY-MM-DDTHH:MM:SS；相对时间 5d/1h/10min/30s

- `-t` --timestamps 显示日志时间戳

#### `port`

查询单个容器的端口映射关系，基本语法 `docker port CONTAINER [PRIVATE_PORT[/PROTO]]`


#### `rm`

删除容器，基本语法 `docker rm [OPTIONS] CONTAINER [CONTAINER...]`  
> `docker container prune` 批量清理所有停止状态（exited 状态）容器

Options

- `-f` --force 强制删除
- `-v` --volumes 删除容器，并同时清理该容器关联的所有匿名数据卷（命名卷不删除）

#### `export`

将容器文件系统导出为原始 tar 归档文件

#### `import`

#### `cp`

#### `commit`

### 多容器相关命令

多容器应用批量管理工具，实现一键「启动 / 停止 / 重启 / 销毁」整个应用集群，基本语法 `docker compose [OPTIONS] COMMAND`

Options

- `-f compose_file` --file，指定compose配置文件


- docker compose使用yaml文件管理多个容器，协同工作

#### `up`

基本语法 `docker compose up [OPTIONS] [SERVICE...]`

Options

- `-d` --detach，后台运行

#### `down`

#### `ps`

### 网络管理

Docker 网络管理，基本语法 `docker network COMMAND`

- docker network(bridge), host, none, docker network list


#### `ls`

- `ls` 列出主机上所有 Docker 网络（默认和自定义网络）

#### `inspect`

- `inspect` 查看指定网络的详细信息（容器关联、IP 段等）
- `docker inspect CONTAINER | jq -c .[0] | jq -c .Mounts` 获取指定容器卷信息
    > 修改挂载卷信息：`备份数据 -> 删除旧容器 -> 用新挂载方式启动`
    > `docker cp CONTAINER:/path/to/volume/. /path/to/dump`

#### `create`

#### `connect`

#### `disconnect`

#### `rm`

#### `prune`

### 仓库操作 

1. `docker pull docker.io/library/image_name:tag` 拉取镜像，library为命名空间
    - `--platform` 指定拉取镜像的运行架构
2. `docker tag image_name:tag_name new_image_name:new_tag_name` 重命名docker镜像，常搭配`pull + tag + rmi` 实现从代理hub中下载镜像