---
title: "pandas"
---

```python
import pandas as pd
```

### Series 构建

```python
# 通过 list 构建
s = pd.Series([1, 2, 3, 4])

# 通过 dict 构建（key 作为 index）
s = pd.Series({'a': 1, 'b': 2, 'c': 3})

# 指定 index
s = pd.Series([1, 2, 3], index=['x', 'y', 'z'])

# 指定 name
s = pd.Series([1, 2, 3], name='my_series')
```

#### Series 属性与方法

```python
s.values          # 获取数组值
s.index           # 获取索引
s.name            # 获取名称

s.head(n=5)       # 前 n 行
s.tail(n=5)       # 后 n 行
s.describe()      # 描述性统计
```

### DataFrame 构建与存储

=== "构建"
    ```python
    ''' 通过 dict 构建 '''
    d = {'col_name1': [1, 2], 'col_name2': [3, 4]}
    df = pd.DataFrame(data=d)

    ''' 通过 多维数组 构建 '''
    d = np.array([
        [1, 2, 3],              # 1-row
        [4, 5, 6],              # 2-row
        [7, 8, 9]               # 3-row
        ])
    df = pd.DataFrame(
        d,                      
        columns=['a', 'b', 'c'] # 指定列名
        )

    ''' read csv file '''
    df = pd.read_csv(
        filepath_or_buffer, 
        sep="\t",               # 指定分隔符
        encoding="utf-8"
        )

    ''' read json file '''
    df = pd.read_json(
        filepath_or_buffer,     # 要求整个文件为 json 格式，而不是单行 json
        encoding="utf-8"
    )
    ```

=== "保存"
    ```python
    df.to_csv(
        path_or_buf=None,       # 存放路径
        sep=",",                # 指定分隔符
        index=True,             # 是否保存行索引
        encoding=None,          # 指定编码，可能存在 utf-8-sig 与 utf-8 的解码差异
        )

    df.to_json(
        
    )
    ```

#### df 属性

=== "行相关"
    ```python
    df.index                # 返回 sub_df 中对应 entire_df 所有行的下标
    df.iloc[idx]            # 按行索引（从 0 开始）
                            # 使用 sub_df.index 访问时应通过 entire_df[idx] 获取
    df.loc[idx]             # 获取 sub_df 中 idx 号数据
    df.loc[idx1, idx2]      # 获取 sub_df 中 idx1, idx2 号数据
    df.loc[start:end]       # 切片 sub_df 中 [start, end) 区间内号数据
    ```

=== "列相关"
    获取列信息，如列名、指定列、列集合、列切片等
    ```python
    df.columns              # 返回 df 的列信息
    df[column_name]         # 返回列 column_name
    df[:, col1]             # 获取列 col1 数据
    df[:, [col1, col2]]     # 获取列 col1, col2 数据
    df[:, col1:col5]        # 获取列 [col1, col5] 数据
    ```

=== "元素相关"
    ```python
    df.loc[idx, column_name] = assign_value
                            # 通过行、列对定位元素并进行赋值
    ```

#### df 方法

=== "整体相关"
    ```python
    df.notna/notnull()      # 返回 df 中各数值不为空值情况
    df.isna/isnull()        # 返回 df 中各数值为空值情况
    ```

=== "行相关"
    ```python
    df.iterrows()           # 等价于 zip(df.index, df.rows)
    ```

=== "列相关"
    ```python
    df[column_name].unique()# 返回列 column_name 的值域
    df[column_name].value_counts(
        normalize=False,    # {False: 频数; True: 频率}
        sort=True,          # 是否排序
        ascending=False,    # 是否升序显示
        bins=None,          # Union(int, list[int]), 指定统计区间，前者设定区间数，后者直接设定区间边界
        dropna=True         # 是否忽略空值统计
    )
    ```

=== "元素相关"
    ```python
    pd.notna/notnull(obj)   # 返回输入 obj 各数值不为空情况
    pd.isna/isnull(obj)     # 返回输入 obj 各数值为空情况
    ```

#### 数据筛选

=== "级联过滤"
    ```python
    # 各筛选条件用 `()` 分割，逻辑操作符与&、或|、非~
    df[
        ~(df[column_name_1] == value2) &
        (df[column_name_2] == value2)
    ]

    # 多值筛选
    df[column_name].isin([value1, value2, value3])

    df.query(expr)
    ```

#### apply 与 applymap

=== "Series.apply"
    ```python
    s.apply(func)           # 对 Series 每个元素应用函数
                            # func: 函数或 lambda 表达式
    s.applymap(func)        # 已废弃，DataFrame 专用
    ```

=== "DataFrame.apply"
    ```python
    df.apply(func, axis=0)  # 沿列方向应用函数（对每列操作）
    df.apply(func, axis=1)  # 沿行方向应用函数（对每行操作）
                            # axis: {0: 列方向; 1: 行方向}
                            # func: 函数或 lambda 表达式
    ```

=== "DataFrame.applymap"
    ```python
    df.applymap(func)       # 对 DataFrame 每个元素应用函数
                            # func: 函数或 lambda 表达式
    ```

=== "示例"
    ```python
    # Series.apply 示例
    s = pd.Series([1, 2, 3, 4])
    s.apply(lambda x: x * 2)        # [2, 4, 6, 8]
    s.apply(np.sqrt)                # [1.0, 1.414, 1.732, 2.0]

    # DataFrame.apply 示例
    df = pd.DataFrame({'a': [1, 2], 'b': [3, 4]})
    df.apply(np.sum, axis=0)        # 每列求和：a=3, b=7
    df.apply(np.sum, axis=1)        # 每行求和：[4, 6]
    df.apply(lambda row: row['a'] + row['b'], axis=1)  # 每行 a+b

    # DataFrame.applymap 示例
    df.applymap(lambda x: x * 2)    # 每个元素乘以 2
    df.map(lambda x: x > 2)         # Series 元素映射
    ```
