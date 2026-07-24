### 文件操作

#### 文件信息查看

- 信息查看：`ls`
- 内容统计：`wc`

#### 文件权限操作

- 修改权限：[chmod](./file_related/permission_related.md#chmod)
- 修改文件所有者：[chown](./file_related/permission_related.md#chown)

#### 文件内容查看

- 原始文本：`less/more`、`head/tail` tail -f 
- 内容过滤：[`grep`](file_related/grep.md#grep)
- 内容比较：[`diff`](file_related/diff)

#### 数据格式处理

- json：[`jq`](file_related/jq.md)
- xlsx：[`xlsx2csv`]()
- pandoc：[`pandoc`](file_related/pandoc.md)

#### 文件修改

- 信息修改：`mv`、`cp`、`rm`、`mkdir`
- 内容连接：`cat` (文件末无\n会同行连接)
- 内容合并：`join`类似于数据库join命令、`paste`
- 内容分割：`cut`：被awk覆盖了, [`awk`](file_related/awk.md)
- 内容拆分：[`split`](file_related/split)
- 内容去重：[`sort`](file_related/sort.md) + [`uniq`](file_related/uniq.md)
- 内容替换：[`tr`](file_related/tr.md)、[`sed`](file_related/sed.md)
- 内容扰动：`shuf`
- 文件传输：[`nc`](file_related/nc.md)
- [输入输出重定向](file_related/redirection.md)：`<`、`>(>)`

### 数值操作

- format输出：[`printf`](numeric_operation/printf.md)，[`echo`](numeric_operation/echo.md)
- 序列相关：[`seq`](numeric_operation/seq.md)
- [循环操作]((numeric_operation/condition_cmd.md))：`for`、`while`、`until`

### 软件安装

#### 下载工具

- `rpm`、`yum`、[`apt(-get)`](./download_related/apt.md)
- `dpkg`

### 进程相关

- 查看进程: [`top`](process_related/top.md)、`ps`
- 终止进程: `kill`
- watch -n 1

#### 进程调度

- 多线程：[并发](process_scheduling/concurrent.md)、[并行](process_scheduling/parallel.md)

### 网络操作

- [`curl`](network_operation/curl.md)
- [`wget`](network_operation/wget.md)
- [`ping`](network_operation/ping.md)
- [`netstat`](network_operation/netstat.md)
- [`telnet`](network_operation/telnet.md)
- [`ufw`](network_operation/ufw.md)