### tf.Operation

节点Operation: name + op + input + attrs  
边Tensor：

session=run operation + eval tensor


TensorBoard
```python
tf.summary.FileWriter(logdir, graph=None, max_queue=10, flush_secs=120, graph_def=None, filename_suffix=None, session=None) 
logdir，日志路径 
graph， 
filename_suffix，日志文件后缀 
```