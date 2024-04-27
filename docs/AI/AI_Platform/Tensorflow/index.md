Tensorflow的功能是定义并初始化一个计算图，通过`sess.run()`来执行这个计算图。

- [Dataset](data_fetch/Dataset.md)
- [warm_up]：
```python
is_warmup = tf.cast(global_steps_int < warmup_steps_int, tf.float32)
learning_rate = ((1.0 - is_warmup) * learning_rate + is_warmup * warmup_learning_rate)
optimizer = tf.train.AdamOptimizer(learning_rate)
tvars = tf.trainable_variables()
grads = tf.gradients(loss, tvars)
(grads, _) = tf.clip_by_global_norm(grads, clip_norm=1.0)
train_op = optimizer.apply_gradients(list(zip(grads, tvars)), global_step=global_step)

sess.run(train_op)
```