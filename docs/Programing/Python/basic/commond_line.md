
#### -c 
允许你直接在命令行中执行一段 Python 代码，多行代码可以通过分号 ; 分隔，或者使用三引号 ''' 或 """ 来包含多行字符串。
```python
python -c "import tensorflow as tf; print(tf.test.is_gpu_available())"
```
#### -u