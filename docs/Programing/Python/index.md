### 基础部分

1. [解释器命令行选项](basic/commond_line.md)
1. [`class`](basic/class.md) 类
1. [内置方法](basic/builtins.md)
1. 静态类型注解

### 细节部分

1. [`Container`](details/Container.md) 容器类，包括Iterable、Iterator和Generator
1. [`lambda`](details/lambda.md) 匿名函数 
1. [`Decorator`](details/Decorator.md) 修饰符
1. [`fstring`](details/fstring.md) 字符串插值

### 自用库

#### 算法类

- [`Aho-Corasick`](libs/ahocorasick.md)：A-C算法，多模式匹配中的经典算法
- [`collections`](libs/collections.md)：Python内建的一个集合模块，提供了许多有用的集合类和方法。

#### 功能类

- [`threading`](libs/concurrent_programming.md#threading)、[`multiprocessing`](libs/concurrent_programming.md#multiprocessing)、[`asyncio`](libs/concurrent_programming.md#asyncio)、[`concurrent.futures`](libs/concurrent_programming.md#concurrentfutures)：并发编程（多线程，多进程，异步编程）
- [`functools`](libs/functools.md)：提供一些高阶函数
- [`itertools`](libs/itertools.md)
- [`os`](libs/os.md)：提供一些方便使用操作系统相关功能的函数
- [`sys`](libs/sys.md)：提供一些方便Python解释器
- [`atexit`]
- [`argparse`](libs/argparser.md#argparse)、[`tf.flags`](libs/argparser.md#tfflags)：接受从终端传入的命令行参数
- [`tqdm`](libs/tqdm.md)：Iterable的遍历进度显示库


#### 统计、绘图

- [`matplotlib`](libs/matplotlib.md)：提供数据绘图功能的第三方库
- [`wordlcoud`](libs/wordcloud.md)：绘制词汇组成类似云的彩色图形
  
#### 资源库

- [`emoji`](libs/emoji.md)：提供了一些emoji的相关操作，emoji收集新而全
- [`OpenCC`](libs/opencc.md)：包含中文繁简体转换库
- [`pypinyin`](libs/pypinyin.md)：汉字转拼音的库
- [`unicodedata`](libs/unicodedata.md)
- [`codecs`](libs/codecs.md)
- [`googletrans`](libs/googletrans.md)：google翻译api

#### utils
- [`ahocorasick.py`](utils/ahocorasick.md)
- [`generate_regrex`](utils/generate_regrex.md)
- [`char_alpha_numeric.py`](utils/char_alpha_numeric.md)
- [`random_methods.py`](utils/random_methods.md)
- [`general_dataset_utils.py`](utils/general_dataset_utils.md)、[`torch_dataset_utils.py`](utils/torch_dataset_utils.md)、[`tf1_dataset_utils.py`](utils/tf1_dataset_utils.md)
- [`tokenization.py`](utils/tokenization.md)

### 进阶：文件读写

#### 数据格式

- [`bin`](libs/file_format.md#bin)：二进制文件访存
- [`json`](libs/file_format.md#json)：提供了在JSON数据和Python对象之间进行转换和序列化的功能。
- [`pickle`](libs/file_format.md#pkl)：Python专用自定义存储数据格式
- [`csv`](libs/xlsx.md#csv)：CSV文件读写
- [`xlsx`](libs/xlsx.md#xlsx)：excel文件读写

### 进阶：数据处理

### 进阶：AI Libraries

#### 机器学习

- [`sklearn`](ai_libs/sklearn/sklearn.md)
- [`gensim`](ai_libs/gensim/gensim.md)
- [`lightgbm`]()
- [`fasttext`] pip install fasttext-wheel

#### 分词方法

- BPE 分词器：[`sentencepiece`](ai_libs/bpe_tokenizer.md)、[`tokenization`](ai_libs/bpe_tokenizer.md)
- [`jieba`]()

### 进阶: 爬虫

### 进阶: GUI

1. [`tkinter`]()
1. [`pyqt`]()
