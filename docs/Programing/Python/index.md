### 基础部分

1. [解释器命令行选项](basic/commond_line)
1. [`class`](basic/class) 类
1. [内置方法](basic/builtins)
1. 静态类型注解

### 细节部分

1. [`Container`](details/Container) 容器类，包括Iterable、Iterator和Generator
1. [`lambda`](details/lambda) 匿名函数 
1. [`Decorator`](details/Decorator) 修饰符
1. [`fstring`](details/fstring.md) 字符串插值

### 自用库

#### 算法类

- [`Aho-Corasick`](libs/ahocorasick)：A-C算法，多模式匹配中的经典算法
- [`collections`](libs/collections)：Python内建的一个集合模块，提供了许多有用的集合类和方法。

#### 功能类

- [`threading`](libs/concurrent_programming/#threading)、[`multiprocessing`](libs/concurrent_programming/#multiprocessing)、[`asyncio`](libs/concurrent_programming/#asyncio)、[`concurrent.futures`](libs/concurrent_programming/#concurrent.futures)：并发编程（多线程，多进程，异步编程）
- [`functools`](libs/functools)：提供一些高阶函数
- [`itertools`](libs/itertools.md)
- [`os`](libs/os)：提供一些方便使用操作系统相关功能的函数
- [`sys`](libs/sys)：提供一些方便Python解释器
- [`argparse`](libs/argparser/#argparse)、[`tf.flags`](libs/argparser/#tfflags)：接受从终端传入的命令行参数
- [`tqdm`](libs/tqdm)：Iterable的遍历进度显示库


#### 统计、绘图

- [`matplotlib`](libs/matplotlib)：提供数据绘图功能的第三方库
- [`wordlcoud`](libs/wordcloud)：绘制词汇组成类似云的彩色图形
  
#### 资源库

- [`emoji`](libs/emoji)：提供了一些emoji的相关操作，emoji收集新而全
- [`OpenCC`](libs/opencc)：包含中文繁简体转换库
- [`pypinyin`](libs/pypinyin)：汉字转拼音的库
- [`unicodedata`](libs/unicodedata)
- [`codecs`](libs/codecs.md)
- [`googletrans`](libs/googletrans.md)：google翻译api

#### utils
- [ahocorasick.py](utils/ahocorasick)
- [char_alpha_numeric.py](utils/char_alpha_numeric)
- [random_methods.py](utils/random_methods)

### 进阶：文件读写

#### 数据格式

- [`bin`](libs/file_format/#bin)：二进制文件访存
- [`json`](libs/file_format/#json)：提供了在JSON数据和Python对象之间进行转换和序列化的功能。
- [`pickle`](libs/file_format/#pkl)：Python专用自定义存储数据格式
- [`csv`](libs/xlsx/#csv)：CSV文件读写
- [`xlsx`](libs/xlsx/#xlsx)：excel文件读写

### 进阶：数据处理

### 进阶：AI Libraries

#### 机器学习

- [`sklearn`](ai_libs/sklearn/sklearn)
- [`gensim`](ai_libs/gensim/gensim)
- [`lightgbm`]()

#### 分词方法

- BPE 分词器：[`sentencepiece`](ai_libs/bpe_tokenizer.md)、[`tokenization`](ai_libs/bpe_tokenizer.md)
- [`jieba`]()

### 进阶: 爬虫

### 进阶: GUI

1. [`tkinter`]()
1. [`pyqt`]()
