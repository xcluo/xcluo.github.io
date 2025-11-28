---
title: "pandas"
---

```python
import pandas as pd
```
### DataFrame构建与存储

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
        filepath_or_buffer,     # 要求整个文件为json格式，而不是单行json
        encoding="utf-8"
    )
    ```

=== "保存"
    ```python
    df.to_csv(
        path_or_buf=None,       # 存放路径
        sep=",",                # 指定分隔符
        index=True,             # 是否保存行索引
        encoding=None,          # 指定编码，可能存在utf-8-sig与utf-8的解码差异
        )

    df.to_json(
        
    )
    ```


#### df属性
=== "行相关"
    ```python
    df.index                # 返回sub_df中对应entire_df所有行的下标
    df.iloc[idx]            # 按行索引（从0开始）
                            # 使用sub_df.index访问时应通过entire_df[idx]获取
    df.loc[idx]             # 获取sub_df中idx号数据
    df.loc[idx1, idx2]      # 获取sub_df中idx1, idx2号数据
    df.loc[start:end]       # 切片sub_df中[start, end)区间内号数据
    ```

=== "列相关"

    ```python
    df.columns              # 返回df的列信息
    df[column_name]         # 返回列column_name
    df[:, col1]             # 获取列col1数据
    df[:, [col1, col2]]     # 获取列col1, col2数据
    df[:, col1:col5]        # 获取列[col1, col5]数据
    ```

=== "元素相关"
    ```python
    df.loc[idx, column_name] = assign_value
                            # 通过行、列对定位元素并进行赋值
    ```


#### df方法
=== "整体相关"
    ```python
    df.notna/notnull()      # 返回df中各数值不为空值情况
    df.isna/isnull()        # 返回df中各数值为空值情况
    ```

=== "行相关"
    ```python
    df.iterrows()           # 等价于 zip(df.index, df.rows)
    ```

=== "列相关"
    ```python
    df[column_name].unique()# 返回列column_name的值域
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
    pd.notna/notnull(obj)   # 返回输入obj各数值不为空情况
    pd.isna/isnull(obj)     # 返回输入obj各数值为空情况
    ```


#### 数据筛选
=== "级联过滤"
    ```python
    # 各筛选条件用`()`分割，逻辑操作符与&、或|、非~
    df[ 
        ~(df[column_name_1] == value2) &
        (df[column_name_2] == value2) 
    ]

    # 多值筛选
    df[column_name].isin([value1, value2, value3])

    df.query(expr)
    ```


