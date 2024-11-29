#### input
1. `< log`  
重定向输入，功能和效果可省略，只是为了更明确地表示输入重定向，即`cat log` 等价于 `cat < log`

2. `<<`  
3. `<<<`

#### output
1. `>(>) log`  
    覆盖(追加)输出，即将stdout结果以覆盖(追加)方式写入目标文件  
2. `&>(>) log`  
    将stdout和stderr一起覆盖(追加)写入目标文件
3. `>&`  
    将一个文件描述符对应的结果追加至另一个文件描述符对应结果中作为整体，如`2>&1`和`1>&1`
4. `1>(>) log`  
    等价于 `>(>) log`，即将stdout以覆盖(追加)方式写入目标文件  
5. `2>(>) log`  
    将stderr以覆盖(追加)方式写入目标文件  
!!! info ""
    - file operator 1：stdout，即standard output
    - file operator 2：stderr，即standard error output
    - 功能上 `2>&1 > log` 等价于 `&> log`