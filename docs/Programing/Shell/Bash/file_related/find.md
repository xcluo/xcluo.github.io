---
title: "find"
---

核心功能是在指定目录及其所有子目录中，根据各种条件（如文件名、大小、类型等）递归地搜索文件，并对结果执行相应操作。基本语法为 `find [PATH] [EXPRESSION...]`

> 若不指定路径，默认为当前目录 `.`  
> 若不指定动作，默认为 `print`（打印匹配路径）

### 常用条件

#### 按名称查找

| 选项 | 描述 |
| --- | --- |
| `-name` | 按文件名匹配（区分大小写） |
| `-iname` | 按文件名匹配（忽略大小写） |
| `-regex` | 按正则表达式匹配完整路径 |

```bash
find /path -name "*.md"              # 查找所有 .md 文件
find /path -iname "*.log"            # 忽略大小写查找 .log 文件
find /path -name "file?*".txt        # 通配符匹配
```

#### 按类型查找

| 选项 | 描述 |
| --- | --- |
| `-type f` | 普通文件 |
| `-type d` | 目录 |
| `-type l` | 符号链接 |

```bash
find /path -type d                   # 查找所有目录
find /path -type f                   # 查找所有普通文件
```

#### 按大小查找

```bash
find /path -size +100M               # 大于 100MB 的文件
find /path -size -50K                # 小于 50KB 的文件
find /path -size 10M                 # 等于 10MB 的文件
```
> 单位：`c`（字节）、`k`（KB）、`M`（MB）、`G`（GB）  
> `+` 表示大于，`-` 表示小于，无符号表示等于

#### 按时间查找

| 选项 | 描述 |
| --- | --- |
| `-mtime` | 按修改时间（天为单位） |
| `-atime` | 按访问时间（天为单位） |
| `-ctime` | 按状态改变时间（天为单位） |

```bash
find /path -mtime -7                  # 7 天内修改过的文件
find /path -mtime +30                 # 30 天前修改过的文件
find /path -atime -1                  # 1 天内访问过的文件
```
> `+n` 表示 n 天前，`-n` 表示 n 天内，`n` 表示恰好 n 天前

#### 按权限查找

```bash
find /path -perm 755                  # 权限恰好为 755 的文件
find /path -perm -755                 # 权限包含 755 的文件（至少具有这些权限）
find /path -perm /755                 # 权限与 755 有任意匹配的文件
```

#### 按所有者查找

```bash
find /path -user username             # 按用户名查找
find /path -uid 1000                  # 按用户 ID 查找
find /path -group groupname           # 按组名查找
find /path -gid 1000                  # 按组 ID 查找
```

### 常用动作

| 选项 | 描述 |
| --- | --- |
| `-print` | 打印匹配路径（默认动作） |
| `-print0` | 打印路径，以 null 字符结尾（配合 xargs -0 使用） |
| `-ls` | 以 ls -dils 格式打印详细信息 |
| `-delete` | 删除匹配的文件 |
| `-exec cmd {} \;` | 对每个匹配项执行命令 |
| `-exec cmd {} +` | 批量执行命令（效率更高） |
| `-ok cmd {} \;` | 同 -exec，但执行前需确认 |
| `-mtime -n -delete` | 组合条件与动作 |

### 组合条件

```bash
# 与关系（-o 表示或，-a 表示与，可省略）
find /path -name "*.md" -type f       # 名为 *.md 的普通文件
find /path -name "*.md" -o -name "*.txt"  # 名为 *.md 或 *.txt 的文件
find /path \( -name "*.md" -o -name "*.txt" \) -type f  # 括号需转义

# 否定
find /path ! -name "*.log"            # 排除 .log 文件
find /path -not -name "*.log"         # 同上
```

### 实用示例

```bash
# 查找并删除 7 天前的 log 文件
find /var/log -name "*.log" -mtime +7 -delete

# 查找大文件并按大小排序
find /path -type f -size +100M -exec ls -lh {} \; | sort -k5 -h

# 查找文件并统计数量
find /path -name "*.md" | wc -l

# 查找并复制文件
find /path -name "*.conf" -exec cp {} /backup/ \;

# 查找并执行批量操作（效率更高）
find /path -name "*.tmp" -exec rm {} +

# 配合 xargs 使用
find /path -name "*.jpg" -print0 | xargs -0 -P 4 convert {} thumbnail_{}

# 查找空文件/空目录
find /path -type f -empty              # 空文件
find /path -type d -empty              # 空目录

# 查找最近修改的文件
find /path -type f -mtime -1 -printf "%T+ %p\n" | sort
```

### 注意事项

1. **权限问题**：需要足够的权限才能访问目标目录
2. **性能考虑**：`-exec cmd {} +` 比 `-exec cmd {} \;` 效率更高
3. **特殊字符**：文件名含空格或特殊字符时，使用 `-print0` + `xargs -0`
4. **括号转义**：组合条件时，括号需要用 `\` 转义
5. **动作顺序**：多个动作时，注意执行顺序（如 `-print -delete`）
