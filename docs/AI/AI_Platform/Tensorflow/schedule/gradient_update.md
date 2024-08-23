1. (可选项) learning rate schedule
2. (可选项) warmup
3. 计算参数的反向梯度
4. (可选项) gradient clipping
5. 选定optimizer
6. 应用optimizer实现gradient update
7. execute operation graph one time

=== "TF 1.x"
    ```python
    tvars = tf.trainable_variables()
    # calculate trainable_variables bp gradients
    grads = tf.gradients(loss, tvars)
    # gradients clip
    (grads, _) = tf.clip_by_global_norm(grads, clip_norm=1.0)
    # choose optimizer
    optimizer = tf.train.AdamOptimizer(learning_rate)
    # update using (trainable_variable, gradient) pairwise
    train_op = optimizer.apply_gradients(zip(grads, tvars), global_step=global_step)
        # global_step: IntegerTensor，每次执行会对该tensor进行加1操作
    # execute operation graph one time
    sess.run(train_op)
    ```

### lr_schedule
#### exponential_decay
learning rate指数衰减

$$\text{decayed_lr} = init\_lr * \text{pow}(decay\_rate, \frac{global\_step}{decay\_steps})$$

<div class="one-image-container">
    <img src="\AI\AI_Platform\Tensorflow\schedule\images\exponential_decay.jpg" style="width: 50%;">
</div>

=== "TF 1.x"
    ```python
    def exponential_decay(
            learning_rate,              # init_lr
            global_step,                # 当前训练的step计数
            decay_steps,                # 用于计算衰减指数
            decay_rate,                 # 幂函数底，一般<1，越小衰减越快
            staircase=False,            # lr是否离散衰减
                                        # staircase=False, 连续衰减，每步衰减一次
                                        # staircase=True, 离散衰减，每decay_rate步衰减一次
            name=None):
    ```

#### polynomial_decay
learning rate多项式衰减

$$\text{decayed_lr} = (init\_lr - end\_lr) * \text{pow}(1-\frac{global\_step}{decay\_steps}, {power}) + end\_lr $$

<div class="one-image-container">
    <img src="\AI\AI_Platform\Tensorflow\schedule\images\polynomial_decay.jpg" style="width: 50%;">
</div>

=== "TF 1.x"
    ```python
    def polynomial_decay(
            learning_rate,              # init_lr
            global_step,                # 当前训练的step计数
            decay_steps,                # 用于计算衰减底数
            end_learning_rate=0.0001,   # end_lr, 允许衰减至的最小lr
            power=1.0,                  # 多项式幂函数指数，一般≥1，越大衰减越快
            cycle=False,                # 受否周期衰减
                                        # cycle=False，cal_global_step=min(global_step, decay_steps)
                                        # 底数 -> `(1 - cal_global_step/decay_steps)`
                                        # cycle=True, cal_decay_steps=decay_steps * ceil(global_step/decay_steps)
                                        # 底数 -> `(1 - global_step/cal_decay_steps)`
            name=None):
    ```

#### natural_exp_decay
#### inverse_time_decay
#### consine_decay
#### linear_consine_decay
#### noise_linear_consine_decay

### warmup
在模型训练初期对(衰减后的)learning rate执行warmup操作

=== "TF 1.x"
    ```python
    ''' step 处于 [0, num_warmup_steps] 区间执行warmup操作 '''
    global_steps_int = tf.cast(global_step, tf.int32)
    warmup_steps_int = tf.constant(num_warmup_steps, dtype=tf.int32)

    global_steps_float = tf.cast(global_steps_int, tf.float32)
    warmup_steps_float = tf.cast(warmup_steps_int, tf.float32)

    warmup_percent_done = global_steps_float / warmup_steps_float
    warmup_learning_rate = init_lr * warmup_percent_done
    # 通过比较 global_step 和 num_warmup_steps 决定是否需要应用warmup操作
    is_warmup = tf.cast(global_steps_int < warmup_steps_int, tf.float32)
    # 除线性warmup外，还可自定义其它warmup策略
    learning_rate = ((1.0 - is_warmup) * learning_rate + is_warmup * warmup_learning_rate)
    ```


### calculate bp gradients

#### using function
=== "TF 1.x"
    ```python
    ''' tf.gradients(loss, tf.trainable_variables()) '''
    def gradients(
            ys,
            xs,
            grad_ys=None,
            name="gradients",
            colocate_gradients_with_ops=False,
            gate_gradients=False,
            aggregation_method=None,
            stop_gradients=None,
            unconnected_gradients=UnconnectedGradients.NONE):
    ```
#### using optimizer
=== "TF 1.x"
    ```python
    ''' optmizer.compute_gradients(loss, tf.trainable_variables()) '''
    def compute_gradients(self, 
            loss, 
            var_list=None,
            gate_gradients=GATE_OP,
            aggregation_method=None,
            colocate_gradients_with_ops=False,
            grad_loss=None):
    ```
### gradient clipping
梯度裁剪一般用于解决 梯度爆炸(gradient explosion) 问题

#### clip_by_value
将梯度直接修剪为指定区间 `[clip_value_min, clip_value_max]` 内的值
=== "TF 1.x"
    ```python
    def clip_by_value(
            t, 
            clip_value_min, 
            clip_value_max,
            name=None):
    ```
#### clip_by_norm
通过控制梯度的最大范式`t * clip_norm / max(clip_norm, l2norm(t))`对梯度进行约束裁剪
=== "TF 1.x"
    ```python
    def clip_by_norm(
            t, 
            clip_norm, 
            axes=None, 
            name=None):
    ```


### optimizer

#### compute_gradients
使用optimizer计算梯度，见[using optimizer](#using-optimizer)
#### apply_gradients
实现trainable_variables的梯度更新

=== "TF 1.x"
    ```python
    def apply_gradients(self, 
        grads_and_vars,         # list[(gradient, variable)]
        global_step=None,       # 记录训练的step计数张量，每次执行会进行+1操作
        name=None)
    ```
    > 自行声明：`global_step=tf.Variable(0, trainable=False)`  
    > 内置方法声明：`global_step=tf.train.get_or_create_global_step()`

#### minimize
`minimize = compute_gradients + apply_gradients`