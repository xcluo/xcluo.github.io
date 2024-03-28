
### 字符相关
#### 空白字符

=== "Python"
    ```python
    white_space_pattern = re.compile(
        "["
        "\u0000-\u001f"        # ASCII 0~31 控制字符或通信专用字符
        "\\s\u0020\u3000"      # 全半角空格 + \\s
        "\u007f-\u00a0"        # DEL-NBSP
        "\u034f"               # 组合用字形连接符
        "\u2000-\u200f\u2011\u2028-\u202f\u205f-\u206f"
                               # NQSP-右至左标记 + BN_ + LSEP-NNBSEP + MMSP-名义数字形状
        "\ufe00-\ufe0f"        # 变体选择器
        "\U000e0100-\U000e01ef"# 变体选择器补充
        "\ufeff"               # ZWNBSP
        "\u115f\u1160\u3164\uffa0"   
                               # 朝鲜文初声填充符HCF + 朝鲜文中声填充符HJF + 朝鲜文填充符HF + 朝鲜文半宽填充符HWHF
        "\ufff0-\uffff"        # 特殊字符
        "\U000e0000-\U000e007f"# 标签
        "]+"
    )
    ```

=== "Java"
    ```java
    String WHITE_CHAR =
        "[" +
        "\u0000-\u001f" +
        "\\s\u0020\u3000" +
        "\u007f-\u00a0" +
        "\u2000-\u200f\u2011\u2028-\u202f\u205f-\u206f" +
        "\ufe00-\ufe0f" +
        "\ufeff" +
        "\u115f\u1160\u3164\uffa0" +
        "\ufff0-\uffff" +
        "\uDB40\uDC00-\uDB40\uDC7F" +
        "\uDB40\uDD00-\uDB40\uDDEF" +
        "]+";
    ```
#### 标点符号

=== "Python"
    ```python
    punctuation = re.compile(
        "["
        "–—＿_　〃〃〟〾〿„…‧﹏﹑·｡"
        "\\＼"
        "＋\+－\-＊\*／/＝="
        "＃#＄\$％%＆&＠@〰～〜~＾\^"
        "！!？\?｜\|"
        "：；﹔:;，,。 \.、､"
        "\(\)（）［］\[\]｛｝\{\}｢｣〈〉<>＜＞｟｠《》「」『』【】〔〕〖〗〘〙〚〛"
        "＂｀＇‘’〝〞“”'\"‟‛`"
        "]"
    )
    ```

#### 中日韩CJK字符

=== "Python"
    ```python
    def is_chinese(cp):
        if (cp >= 0x4E00 and cp <= 0x9FFF) or       #
            (cp >= 0x3400 and cp <= 0x4DBF) or      #
            (cp >= 0x20000 and cp <= 0x2A6DF) or    #
            (cp >= 0x2A700 and cp <= 0x2B73F) or    #
            (cp >= 0x2B740 and cp <= 0x2B81F) or    #
            (cp >= 0x2B820 and cp <= 0x2CEAF) or    #
            (cp >= 0xF900 and cp <= 0xFAFF) or      #
            (cp >= 0x2F800 and cp <= 0x2FA1F):      #
            return True
        return False

    def is_japanese(cp):
        if (0x3000 <= cp and cp <= 0x303f or
            0x3040 <= cp and cp <= 0x309f or
            0x30a0 <= cp and cp <= 0x30ff):
            return True
        return False

    def is_korean(cp):
        if (0x1100 <= cp and cp <= 0x11ff or
            0x3130 <= cp and cp <= 0x318f or
            0xac00 <= cp and cp <= 0xd7af or
            0xa960 <= cp and cp <= 0xa97f or
            0xd7b0 <= cp and cp <= 0xd7ff):
            return True
        return False
    ```
=== "Java"
    ```java
    boolean is_chinese_char(int cp) {
        if ((cp >= 0x4E00 && cp <= 0x9FFF) || 
            (cp >= 0x3400 && cp <= 0x4DBF) || 
            (cp >= 0x20000 && cp <= 0x2A6DF) || 
            (cp >= 0x2A700 && cp <= 0x2B73F) || 
            (cp >= 0x2B740 && cp <= 0x2B81F) || 
            (cp >= 0x2B820 && cp <= 0x2CEAF) ||
            (cp >= 0xF900 && cp <= 0xFAFF) || 
            (cp >= 0x2F800 && cp <= 0x2FA1F)) {
                return true;
        }
        return false;
    }

    boolean is_japanese(int cp) {
        if (0x3000 <= cp && cp <= 0x303f ||
            0x3040 <= cp && cp <= 0x309f ||
            0x30a0 <= cp && cp <= 0x30ff) {
            return true;
        }
        return false;
    }

    boolean is_korean(int cp) {
        if (0x1100 <= cp && cp <= 0x11ff ||
            0x3130 <= cp && cp <= 0x318f ||
            0xac00 <= cp && cp <= 0xd7af ||
            0xa960 <= cp && cp <= 0xa97f ||
            0xd7b0 <= cp && cp <= 0xd7ff) {
            return true;
        }
        return false;
    }
    ```



### 文件处理
#### JSON文件去重

=== "Bash"
    ```bash title="json内容去重"
    # 1. 头尾分别插入[ 和 ]，每行插入分割符，形成一个list
    # 2. unique_by只在list中生效
    # 3. 自行指定`file_name` 和 `key_name`
    cat <file_name> | sed -e '1i[' -e '2,$i ,' -e '$a]' | jq -c 'unique_by(.<key_name>) | .[]'  > unique_result.json
    ```

=== "Python"
    ```python title="json内容去重"
    import argparse
    import json
    from tqdm import tqdm


    parser = argparse.ArgumentParser()
    parser.add_argument('-f', '--file_name', type=str, required=True, help='var description')
    args = parser.parse_args()  # 通过`args.var_name` 调用相应参数


    if __name__ == "__main__":
        dump_lines = set()
        with open(args.file_name, 'r', encoding='utf-8') as f, \
                open('./unique_result.json', 'w', encoding='utf-8') as fout:
            for i, line in enumerate(tqdm(f), 1):
                try:
                    line = json.loads(line)
                except:
                    print(line)
                    raise ValueError(f"{i}-th line")
                cnt = line.get('content', line.get('c'))
                if cnt in dump_lines:
                    continue
                dump_lines.add(cnt)
                fout.write(json.dumps(line, ensure_ascii=False) + '\n')
                fout.flush()
                if len(dump_lines) >= 40000000:
                    dump_lines.clear()
    ```




### 实用Bash组合命令
#### 批量kill进程
```bash title="批量kill python pids"
# 1. 选择对应进行信息
# 2. 选取进程信息的第二列(pid)进行批量kill
kill `ps -ef | grep python | grep '<common_process_info>' | awk -F' ' '{print $2}'`
```

#### 批量cat文件内容
```bash title="批量cat文件内容"
# 1. 批量cat以"part_"开头的json文件
cat `ls part_*.json`
```

#### 获取json文件的键的取值范围
```bash
# 1. .key_name获取键的值
# 2. sort + uniq 实现去重相同的连续行
cat <json_file_name> | jq -c .<key_name> | sort | uniq
```