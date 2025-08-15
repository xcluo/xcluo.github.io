### 使用方法
- `sort [OPTION] ... [FILE] ...`

#### OPTION
1. `-f, --ignore-case` 大小写不敏感
2. `-k <start>[.[offset]],<end>[.<offset>] --key`，指定键用于排序，其中offset表示键的字符偏移数
    ```bash
    <!-- 以第1列字符2至字符4为键排序, $1[2:4+1] -->
    sort -k 1.2,1.4 file.txt        
    sort --key=1.2,1.4 file.txt
    ```
3. `-n` 按数值大小升序排序
4. `-r` 降序排序，`-nr` 按数值方式降序排序