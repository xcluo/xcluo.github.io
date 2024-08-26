

#### spatial_dropout
随机对没个 channel 的所有值进行 `dropout`
=== "TF 1.x"
    ```python
    def spatial_dropout_2d(input_tensor, keep_prob=1):
        if keep_prob == 1:
            return input_tensor
        elif keep_prob <= 0 or keep_prob > 1:
            raise ValueError(f"keep_prob should in (0, 1], keep_prob=={keep_prob} is invalid!!!")

        if input_tensor.shape.ndims != 3:
            raise ValueError(f"input_tensor should equal 3, but not {input_tensor.shape.ndims}")

        bs, seq, dim = input_tensor.shape.as_list()
        mask = tf.tile(tf.expand_dims(tf.random_uniform(shape=[bs, dim]) < keep_prob,
                                    axis=1),
                    [1, seq, 1])
        spatial_dropout_tensor = tf.where(mask, input_tensor, tf.zeros_like(input_tensor))
        return tf.multiply(tf.constant(1/keep_prob), spatial_dropout_tensor)
    ```

#### max_pooling_dropout
在 `max_pooling` 前每个值进行 `dropout`
=== "TF 1.x"
    ```python
    # 2d 的具体实现等价于 `tf.nn.dropout`
    def max_pooling_dropout_2d(input_tensor, keep_prob=1, name=None):
        return tf.nn.dropout(input_tenosr, keep_prob=keep_prob, name=name)
    ```