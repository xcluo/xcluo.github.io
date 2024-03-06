
### csv(Comma-Separated Values)
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
import xlrd     # pip install xlrd

workbook = xlrd.open_workbook(<file_name>)
# sheet.rows
# sheet.cols
# sheet.row_values(idx)
sheet = workbook.sheet_by_name(<sheet_name>)
```