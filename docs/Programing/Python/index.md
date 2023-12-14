### 基础部分
1. [`class`](basic/class) 类
1. [`内置方法`](basic/builtins)


### 细节部分
1. [`Container`](details/Container) 容器类，包括Iterable、Iterator和Generator
1. [`lambda`](details/lambda) 匿名函数 
1. [`Decorator`](details/Decorator) 修饰符

### 常用第三方库

#### 算法类

- [`Aho-Corasick`](libs/ahocorasick)：A-C算法，多模式匹配中的经典算法
- [`collections`](libs/collections)：Python内建的一个集合模块，提供了许多有用的集合类和方法。

#### 功能类

- [`threading`](libs/threading)：多线程
- [`tqdm`](libs/tqdm)：Iterable的遍历进度显示库
- [`os`](libs/os)：提供一些方便使用操作系统相关功能的函数
- [`sys`](libs/sys)：提供一些方便Python解释器

#### 参数传递

- [`argparse`](libs/argparser/#argparse)：用于命令项选项与参数解析的模块
- [`tf.flags`](libs/argparser/#tfflags)：用于接受从终端传入的命令行参数

#### 统计、绘图

- [`matplotlib`](libs/matplotlib)：提供数据绘图功能的第三方库
- [`wordlcoud`](libs/wordcloud)：绘制词汇组成类似云的彩色图形
  
#### 资源库

- [`emoji`](libs/emoji)：提供了一些emoji的相关操作，emoji收集新而全
- [`OpenCC`](libs/opencc)：一个开源的中文繁简体转换库

### 进阶：文件读写

#### 数据格式

- [`bin`](libs/file_format/#bin)：二进制文件访存
- [`json`](libs/file_format/#json)：提供了在JSON数据和Python对象之间进行转换和序列化的功能。
- [`pickle`](libs/file_format/#pkl)：Python专用自定义存储数据格式

#### 文件读写

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
