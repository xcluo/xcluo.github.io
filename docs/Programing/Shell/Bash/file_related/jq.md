[jq](https://jqlang.github.io/jq/manual/)：commadline JSON processor

### 安装

#### windows
1. [下载网址](https://jqlang.github.io/jq/download/)：{AMD64: 64位; i386: 32位}
2. `~/.bash_profile` 中设置 `alias jq=jqexe_absoulte_path'`
3. 加载配置 `source ~/.bash_profile`


### 使用方法
`jq [options] filter [file]`

#### `options` 选项
- `-c/--compact-output`：紧凑输出，即把一个JSON对象输出在一行
> 通过紧凑输出处理的行中，取消了键与值之间的空格，即 `"key":"value"`
#### `filter` 过滤器
- `keys`：获取当前对象的键
- `.`：获取当前对象
- `[]`：获取整个数组，支持切片
- `select(condition)`：过滤条件

#### `filter` 实用命令

1. 键值选取、组合
```bash
# {"c", "xxx”, "logits": [xxx, ...], "prob": [xxx, xxx]}

# 数据访问
jq -c .prob | jq .[]                    # 逐层操作，获取list: [prob_1, ..., prob_n]
jq -c .prob[]                           # 直接解析，获取n行prob

# 数值修改
jq -c '.name="luo"'                     # 将键name的值修改为 "luo"
jq -c '.name=.name+"luo"'               # 将键name的值在结尾新增 "luo"


# 数据重组
jq '[.a, .b]'
jq '{"content": .c}'
```

2. 数值选择
```bash
jq -c select(.prob[2] > 0.5)             # 选择概率大于0.5的行
jq -c 'select(.c | tostring | length > 10)'
                                         # 将数字转化为字符串形式再比较
```

3. 字符操作
```bash
# 字符串匹配
jq -c 'select(.c == "lxc")'              # 完全匹配
jq -c 'select(.c | contains("lxc"))'     # 部分匹配

jq -c 'select(.c | length > 200)'        # 选择数据中字段c长度大于200的行
jq -c '.c | select(length > 200)'        # 选择字段c长度大于200的字段
jq -c 'select(.c | length > 200) | .c'   # 等价于↑
jq -c 'select(.c | tonumber > 100)'      # 将字符串转化为数字形式再比较（要求能够表示为数字）
```

4. 与其它命令组合操作
```bash
# 通过`paste`和`awk`命令将 raw_content 与 prob 结果合并为一个数组
# + 过滤相应标签概率大于0.5的行
# + 最终JSON对象单行输出
paste test1.txt result1.txt | awk -F '\t' '{print "[" $1 ", "$2 "]" }' | jq -c 'select(.[1].prob[1] > 0.5)'

# 多条件过滤 and/or
jq -c 'select((.s | tonumber > 0.92) and (.c | length > 10))'


# unique_by(.file_name) 适用于单个json list的关键字去重
# 去重 + 并选择内容长度处于 (0, 200] 区间的样本
jq -c 'unique_by(.c) | .[] | select(.c | lenght > 0 and length <= 200)'

# 对键值c去重
sed -e '1i[' -e '2,$i ,' -e '$a]' | jq -c 'unique_by(.c) | .[]'
```

- tostring
- tonumber