### lr_schedule
#### exponential_decay

#### polynomial_decay

$$\text{decayed_learning_rate} = \text{learning_rate} * \text{decay_rate} ^ \frac{\text{global_step}}{\text{decay_steps}}$$

```python
def exponential_decay(
        learning_rate,
        global_step,
        decay_steps,
        decay_rate,
        staircase=False,
        name=None):
```
#### natural_exp_decay
#### inverse_time_decay
#### consine_decay
#### linear_consine_decay
#### noise_linear_consine_decay
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

### warmup
```python
def create_optimizer(
    loss,                   # updated loss
    init_lr,                # initialized learning rate
    num_train_steps,        # total train steps
    num_warmup_steps,       # warmup steps
    manual_fp16=False):     # 
    global_step = tf.train.get_or_create_global_step()

    learning_rate = tf.constant(value=init_lr, shape=[], dtype=tf.float32)

    learning_rate = tf.train.polynomial_decay(
        learning_rate,
        global_step=global_step,
        decay_steps=num_train_steps,
        end_learning_rate=0.0, 
        power=1.0, 
        cycle=False)
    # learning_rate = tf.train.exponential_decay(learning_rate,
    #                                            global_step=global_step,
    #                                            decay_steps=1000, decay_rate=0.9, staircase=True)
    if num_warmup_steps:
        global_steps_int = tf.cast(global_step, tf.int32)
        warmup_steps_int = tf.constant(num_warmup_steps, dtype=tf.int32)

        global_steps_float = tf.cast(global_steps_int, tf.float32)
        warmup_steps_float = tf.cast(warmup_steps_int, tf.float32)

        warmup_percent_done = global_steps_float / warmup_steps_float
        warmup_learning_rate = init_lr * warmup_percent_done

        is_warmup = tf.cast(global_steps_int < warmup_steps_int, tf.float32)
        learning_rate = ((1.0 - is_warmup) * learning_rate + is_warmup * warmup_learning_rate)
    optimizer = tf.train.AdamOptimizer(learning_rate)
    tvars = tf.trainable_variables()
    grads = tf.gradients(loss, tvars)
    if manual_fp16:
        loss_scale_manager = tf.contrib.mixed_precision.ExponentialUpdateLossScaleManager(
            init_loss_scale=2 ** 32, incr_every_n_steps=1000, decr_every_n_nan_or_inf=2, decr_ratio=0.5)
        optimizer = tf.contrib.mixed_precision.LossScaleOptimizer(optimizer, loss_scale_manager)
    (grads, _) = tf.clip_by_global_norm(grads, clip_norm=1.0)
    train_op = optimizer.apply_gradients(list(zip(grads, tvars)), global_step=global_step)
    return train_op
```
