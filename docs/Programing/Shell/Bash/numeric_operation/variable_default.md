---
title: bash变量缺省值语法
---

## 概述

Bash 变量的缺省值语法是处理**未定义或为空**变量的强大工具，常用于：

- 为环境变量设置默认值（如 `${PORT:-8080}`）
- 强制检查必填参数（如 `${API_KEY:?必须设置}`)
- 条件性构建参数（如 `${DEBUG:+-v}`）

| 语法 | 变量为空时 | 变量有值时 | 修改变量 |
| ------ | ----------- | ----------- | --------- |
| `${var:-default}` | 使用 default | 使用 var 值 | ❌ 否 |
| `${var:=default}` | 赋值 default，返回它 | 使用 var 值 | ✅ 是 |
| `${var:+alternate}` | 使用空 | 使用 alternate | ❌ 否 |
| `${var:?error}` | 报错退出 | 使用 var 值 | ❌ 否 |

=== "`${parameter:-word}`"

    **使用默认值（但不赋值）**，这是最常用的缺省值语法。当变量未定义或为空时，返回默认值；变量有值时，返回变量本身。

    ```bash
    # 变量未定义，使用默认值
    echo "${name:-lxc}"
    # 输出：lxc

    name="Alice"
    echo "${name:-lxc}"
    # 输出：Alice（变量有值，优先使用）

    # 典型用法：设置默认值
    PORT="${OPENCLAW_GATEWAY_PORT:-18789}"
    # 等价于：
    if [ -z "$OPENCLAW_GATEWAY_PORT" ]; then
        PORT=18789
    fi
    ```

    > 注意：这只是**使用**默认值，不会修改变量本身。

=== "`${parameter:=word}`"

    **使用默认值（并且赋值）**，与 `:-` 不同的是，它会将默认值真正赋给变量。常用于初始化配置变量。

    ```bash
    echo "${name:=lxc}"
    # 输出：lxc
    # 同时 name 被赋值为"lxc"

    echo "Hello $name"
    # 输出：Hello lxc
    ```

    > 区别于 `:-`，这会**修改变量**本身。

=== "`${parameter:+word}`"

    **有值才替换（Alternate Value）**，与前两个相反——变量有值时返回替代值，变量为空时返回空。这用于条件性添加内容。

    ```bash
    DEBUG_MODE=1

    # 调试时显示，发布时不显示
    echo "开始处理${DEBUG_MODE:+ [调试模式]}"

    # 条件性设置选项
    MYSQL_HOST="${MYSQL_HOST:+-h $MYSQL_HOST}"
    # 如果 MYSQL_HOST 有值，返回 -h xxx；否则返回空
    ```

    > 这是**反向逻辑**：有值才替换。

=== "`${parameter:?word}`"

    **空则报错退出（Error if Empty）**，如果变量为空，输出自定义错误信息并退出脚本。用于强制检查必填参数。

    ```bash
    # 必须提供 API_KEY
    API_KEY="${API_KEY:?错误：API_KEY 未设置}"

    # 自定义错误信息
    CONFIG_FILE="${CONFIG_FILE:?必须通过 -c 指定配置文件}"
    ```

    > 用于**强制检查必填变量**，脚本开头的参数校验。

## 嵌套与组合

```bash
# 嵌套：先判断是否存在，不存在则用默认值
DB_HOST="${DB_HOST:-${DEFAULT_DB_HOST:-localhost}}"

# 组合：使用默认值，同时检查是否为空
FILE_PATH="${FILE_PATH:?必须指定文件路径}"

# 配合命令替换
LOG_DIR="${LOG_DIR:-$(pwd)/logs}"
```

---

## 常见实战用法

```bash
# 1. 环境变量默认值
export NODE_ENV="${NODE_ENV:-development}"
export PORT="${PORT:-3000}"

# 2. 必填参数校验
API_KEY="${API_KEY:?请设置 API_KEY 环境变量}"

# 3. 条件性添加参数
SSH_OPTS="${SSH_OPTS:+-o $SSH_OPTS}"

# 4. 配置文件路径
CONFIG="${CONFIG:-$HOME/.myapp.conf}"

# 5. 调试标志
TRACE="${TRACE:+-x}"  # 有 TRACE 变量则开启调试
bash $TRACE script.sh
```
