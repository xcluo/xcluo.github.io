
### csv
Comma-Separated Values
```python
import csv

with open(<file_name>, 'r', encoding='utf-8') as f:
    reader = csv.reader(f)      # file_iterator
    for line in reader:
        ...
```


### xlsx
```python
import xlwt     # pip install xlwt

workbook = xlwt.Workbook(encoding='utf-8')
sheet = workbook.add_sheet(<sheet_name>, cell_overwrite_ok=True)
sheet.write(row_idx, col_idx, dump_content)
workbook.save(<file_name>)

import xlrd     # pip install xlrd
workbook = xlrd.open_workbook(<file_name>)
sheet = workbook.sheet_by_name(<sheet_name>)
    # sheet.rows：数据表行数
    # sheet.cols：数据表列数
    # sheet.row_values(idx)：数据表idx-th行数据
```