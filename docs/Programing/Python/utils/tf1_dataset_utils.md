```python
import tensorflow as tf
from wheel_utils.general_dataset_utils import *


class DataResource:
    def __init__(
            self,
            data_file,
            tokenizer=None,
            trie=None,
            t2s=None,
            case_sensitive=False,
    ):
        self.data_file = data_file
        self.example_num = self._count()
        self.trie = trie
        self.t2s = t2s
        self.case_sensitive = case_sensitive
        self.tokenizer = tokenizer

    def _count(self):
        with open(self.data_file, 'r', encoding='utf-8') as f:
            for i, _ in enumerate(f, 1):
                pass
        return i

    def __len__(self):
        return self.example_num

    def generator(self):
        f = open(self.input_file, "r", encoding="utf-8")
        for line in f:
            try:
                line = json.loads(line)
            except:
                print(json.dumps(line, ensure_ascii=False))
                raise ValueError

            line = pre_process(self.trie, self.t2s, self.case_sensitive, line)
            inputs = self.tokenizer(line["content"])

            # 生成的tuple每一项都应该是一个list
            yield (
                [line["content"]],
                [line["label"]],
                inputs["input_token_ids"],
                inputs["input_sound_ids"],
                inputs["input_shape_ids"],
                # inputs["input_mask"],
            )

    def next_batch(
            self,
            epochs,
            batch_size,
            shuffle=False,
            drop_remainder=False,
    ):
        dataset = tf.data.Dataset.from_generator(
            self.generator,
            output_types=(                      # 对应参数generator的数据返回类型
                tf.string,
                tf.int32,
                tf.int32,
                tf.int32,
                tf.int32,
            )
        )

        # dataset repeats by epochs times
        dataset = dataset.repeat(epochs)

        # 设置预取池和扰动池大小用于样本随机抽取 #
        # 1. 预取 prefetch_size 个样本进入预取池
        # 2. 当预取池满时，从其中 shuffle_size 个样本中随机抽取一个样本
        #    `idx = random.randint(shuffle_size)` 
        #    `choose prefetch_reservoir[idx]`
        #    `prefetch_reservoir[idx] = current_line`
        # > prefetch_size 和 shuffle_size 越大随机效果越趋近于全局扰动
        if shuffle:
            dataset = dataset.shuffle(buffer_size=100 * batch_size)
        dataset = dataset.prefetch(buffer_size=100 * batch_size)

        dataset = dataset.padded_batch(
            batch_size,
            padded_shapes=(
                [None], [None], [None], [None], [None]
            ),                      # None 表示 `pad_to_longest`, [] 表示不填充
            padding_values=None,    # pad_to_longest
                                    # None 表示缺省PAD {int→0, str→""}
                                    # [PAD] idx≠0 时可自定义对应的pad_value
                                    # 自定义pad_values时需对齐output_types
            drop_remainder=drop_remainder,
        )

        # 转化为iterator
        iterator = dataset.make_initializable_iterator()
        return iterator.get_next(), iterator.initializer
```


#### data_loader
```python
import tensorflow as tf


# session + config
saver = tf.train.Saver(max_to_keep=3)
config = tf.ConfigProto()
config.allow_soft_placement = True
config.gpu_options.allow_growth = True
# config.gpu_options.per_process_gpu_memory_fraction = .1
sess = tf.Session(config=config)

# prepare dataset
dataset = DataResource("./")
dataset_next_batch, dataset_initializer = dataset.next_batch()
sess.run(dataset_initializer)           # 每次运行该命令会重新生成dataset_generator
                                        # 因此可重复valid_dataset_initializer用于效果验证
while True:
    try:
        generated_batch = sess.run(dataset_next_batch)
    except tf.errors.OutOfRangeError:   # generator遍历完成
        saver.save(sess, dump_model_dir, global_step=cur_step)
```