```python
import tensorflow as tf


class DataResourece:
    def __init__(self, input_file, do_lower_case, shuffle, epochs):
        self.input_file = input_file
        self.shuffle = shuffle
        self.epochs = epochs
    
    def generator(self):
        f = codecs.open(self.input_file, "r", encoding="utf-8")
        for line in f:
            # 数据读取 + 处理
            ## example为类对象，feature为dict，需确保其内ids长度一致
            yield [content], [label], example.features["input_ids"], example.features["input_mask"], example.features[
                "segment_ids"], example.features["label_ids"]
    
    def next_batch(self, batch_size):
        dataset = tf.data.Dataset.from_generator(
            self.generator, 
            output_types=(tf.string, tf.string, tf.int32, tf.int32, tf.int32, tf.int32)   # 对应generator返回的数据类型
            )
        dataset = dataset.repeat(self.epochs)
        if self.shuffle:
            #
            dataset = dataset.shuffle(buffer_size=100 * batch_size)
        
        dataset = dataset.prefetch(buffer_size=100 * batch_size)
        dataset = dataset.padded_batch(
            batch_size, 
            padded_shapes=([None], [None], [None], [None], [None], [None])
        )
        # iterator = dataset.make_one_shot_iterator()
        iterator = dataset.make_initializable_iterator()
        return iterator.get_next(), iterator.initializer
```