### 分布
#### Normal Distribution
$X \sim\mathcal{N}(\mu, \sigma^2)$，密度函数$f(x)=\frac{1}{\sqrt{2\pi}\sigma}e^{-\frac{(x-\mu)^2}{2\sigma^2}}$
- 独立分布正态分布组合 $Z = aX  + bY$：$E(Z)=a\mu_X + b\mu_Y$ 和 $Var(Z) = a^2\sigma^2_X + b^2\sigma_Y^2$
- 累计分布函数CDF $\Phi(x)=Pr(X\le x)=\frac{1}{\sqrt{2\pi}}\int_{-\infty}^{x}e^{-\frac{x^2}{2}}dx$

#### Poisson Distribution
泊松分布，是一种离散概率分布，用于描述在一定时间或空间内事件发生的次数。它特别适用于当这些事件是独立发生且以恒定的平均速率出现时的情景。

#### Zipfian Distribution
一个离散幂律概率分布，也就是常常提到的长尾分布，即某个对象的出现频率与其排名成反比（少数对象占据了大部分频率，大多数对象的频率很低），通常用于描述自然语言、城市人口、网站访问量等领域的现象。

$$
f(k) = \frac{C}{k^s}
$$

> $k$ 为对象的排名，$C$ 为归一化常数，$s$ 为分布的幂律超参，通常接近于1

### 采样方式
#### Importance Sampling
- 目标：估计目标分布$P(x)$下的期望值

    $$
    \mathbb{E}_{P}[f(x)] = \int f(x)P(x) dx
    $$

- 挑战：如果$P(x)$难以直接采样或计算，可以使用重要性采样  
- 方法如下：  
    1. 选择一个易于采样的提议分布$Q(x)$
    2. 从$Q(x)$中采样$x_1, x_2, \dots, x_N$
    3. 计算重要性权重$w(x_i) = \frac{P(x_i)}{Q_{x_i}}$
    4. 估计期望值

        $$
        \mathbb{E}_{P}[f(x)]\approx\mathbb{E}_{Q}[f(x)\frac{P(x)}{Q(x)}] = \frac{1}{N} \sum_{i=1}^{N}f(x_i)w(x_i)
        $$


- 灵活性：可以选择任意易于采样的提议分布$Q(x)$
- 高效性：通过选择合适的Q(x)，可以减少估计的方差

#### Inverse Sampling
- 逆采样inverse sampling是一种可以从概率分布中生成随机样本的计数。特别**适用于离散分布或某些特定的连续分布**，其中累计分布函数CDF是已知的，并且可以方便求逆，逆采样的基本思路是利用分布的累计分布函数CDF来生成随机数，步骤为
    1. 确定累计分布函数$F(x)$  
    2. 从均匀随机分布`uniform(0, 1)`中生成随机数$u$
    3. 使用逆CDF函数$F^{-1}(x)$满足$x=F^{-1}(u)$【此处可以理解x为$F(x_i)\ge u$的最小$x_i$值】，此时x即为生成的随机数样本


#### Rejection Sampling
- 拒绝采样，也叫接受-拒绝采样Acceptance-Rejection Sampling是一种用于从复杂分布中生成样本的技术。当目标分布$P(x)$的**概率密度函数PDF难以直接采样**时，但可以计算其值或比例，那么就可以使用**拒绝采样方法来间接地生成**样本。步骤为：
    1. 选择一个容易采样的proposal distribution提议分布$Q(x)$，要求提议分布$Q(x)$需要覆盖目标分布$P(x)$的定义域X  
    2. 找到一个常数c，使得对于所有x，$cQ(x)\ge P(x)$，保证分布$P(x)$的值域总是在$cQ(x)$之下  
    3. 采样和接受/拒绝
          - 依照提议分布Q中采样得到x_0  
          - 从均匀分布`uniform(0, 1)` 中采样得到 $u$
          - 如果$u\lt\frac{P(x_0)}{cQ(x_0)}$，则接受x_0作为P生成的样本；否则拒绝x_0，重复上述过程直到获得足够多的样本  
    > 常数c需要手动选择并满足条件2

> https://www.zhihu.com/question/38056285/answer/1803920100


- 马尔科夫链的当前状态只依赖于前一状态，即$P(x_t\vert x_1, x_2, \dots, x_{t-1}) = P(x_t\vert x_{t-1})$
    - $\pi$ 为马尔科夫链的平稳分布，满足$\pi P = \pi$
- 基于马氏链的蒙特卡罗方法采样MCMC (Markov Chain & Monte Carlo)
    - Detailed Balance细致平衡$\pi(i)P(i, j)=\pi(j)P(j, i)  \rightarrow \pi(i)Q(i,j)\alpha(i, j)=\pi(j)Q(j,i)\alpha(i, j)$
    - $P(i, j)=Q(i, j)\alpha(i, j) \rightarrow \alpha(i, j)=\min\{\frac{\pi(j)Q(j,i)}{\pi(i)Q(i,j)}, 1\}$
    - 其中Q为状态转移矩阵，也可以是状态转移概率密度函数，$P(A\rightarrow B) = P(B\vert A)$
    - 经过1~t时间burn-in燃烧期的状态变换后，在[t+1, ∞]后进入细致平衡期，进入细致平衡期后开始进行最终采样
    - 采样点间不互相独立，mixing time t可能会很长
- MH采样 Metropolis Hastings
    - norm.pdf，输入x，依照给定的norm超参μ和σ生成对应的y值
    - norm.rvs，依照给定norm超参μ和σ随机采样size个x值
- Gibbs sampling吉布斯采样
    - 联合概率分布转化为条件概率（高维转化为低维），不一定要轮换坐标轴，只需要符合条件概率分布进行采样即可，不拒绝，所有采样均接受
    - 是α=1情况下的MH采样，为MH采样的特殊形式。适用于随机变量X维度非常高的情况，从t到t+1时刻，只改变一个维度的值。状态转移矩阵取得就是目标概率p(X)。

> https://www.cnblogs.com/pinard/p/6645766.html  
> https://pan.baidu.com/s/1EJonwMsvVWvgo1utzWydHQ#list/path=%2Fsharelink90532273-812830912909501%2FLDA%E4%BB%A3%E7%A0%81&parentPath=%2Fsharelink90532273-812830912909501  
> [逆采样、拒绝采样、MH采样、MCMC采样](https://www.bilibili.com/video/BV1ey4y1t7Jb/?spm_id_from=333.337.search-card.all.click&vd_source=782e4c31fc5e63b7cb705fa371eeeb78)  
> https://zhuanlan.zhihu.com/p/94313808  
> https://zhuanlan.zhihu.com/p/95467302  
> https://www.zhihu.com/topic/20683707/top-answers
> https://www.cnblogs.com/feynmania/p/13420194.html
> https://www.zhihu.com/question/38056285/answer/1803920100
> https://zhuanlan.zhihu.com/p/669645171  
> https://www.jianshu.com/p/5c510694c07e  
> https://blog.csdn.net/Galbraith_/article/details/104577253  
> https://blog.csdn.net/weixin_46265255/article/details/120250624  
> https://blog.51cto.com/u_16213652/12201210  

```python
# 初始化
for each document d in corpus:
    for each word w in document d:
        assign a random topic z to word w

# 定义迭代次数
num_iterations = predefined_number

# 开始吉布斯采样迭代过程
for iteration from 1 to num_iterations:
    for each document d in corpus:
        for each word w in document d:
            # 移除当前单词的主题分配
            remove_topic_assignment_from_word(w)
            
            # 计算新主题的概率分布
            for each topic z in topics:
                # 更新 doc_topic[d][k]^t，由计算式可知每行必和为1，P(x_i^t|x excluds x_i)
                # 更新 topic_word[k][n]^t，由计算式可知每行必和为1
                doc_topic_prob[z] = (count_of_words_in_doc_d_with_topic_z + alpha) / 
                                    (total_words_in_doc_d + num_topics * alpha)
                topic_word_prob[z] = (count_of_word_w_in_topic_z + beta) / 
                                     (total_words_in_topic_z + vocab_size * beta)
                
                # 合并文档-主题和主题-单词概率
                prob_distribution[z] = doc_topic_prob[z] * topic_word_prob[z]
            
            # 根据计算出的概率重新为单词分配主题
            new_topic = sample_from(prob_distribution)
            assign_topic_to_word(w, new_topic)

# 参数估计（在最后一次迭代后）
for each document d in corpus:
    for each topic z in topics:
        doc_topic_distribution[d][z] = (count_of_words_in_doc_d_with_topic_z + alpha) / 
                                       (total_words_in_doc_d + num_topics * alpha)

for each topic z in topics:
    for each word w in vocabulary:
        topic_word_distribution[z][w] = (count_of_word_w_in_topic_z + beta) /
                                        (total_words_in_topic_z + vocab_size * beta)

# 输出最终的主题-文档和主题-单词分布
output doc_topic_distribution, topic_word_distribution
```


```python
# coding: utf-8

# # sklearn-LDA

# 代码示例：https://mp.weixin.qq.com/s/hMcJtB3Lss1NBalXRTGZlQ （玉树芝兰） <br>
# 可视化：https://blog.csdn.net/qq_39496504/article/details/107125284  <br>
# sklearn lda参数解读:https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.LatentDirichletAllocation.html
# <br>中文版参数解读：https://blog.csdn.net/TiffanyRabbit/article/details/76445909
# <br>LDA原理-视频版：https://www.bilibili.com/video/BV1t54y127U8
# <br>LDA原理-文字版：https://www.jianshu.com/p/5c510694c07e
# <br>score的计算方法：https://github.com/scikit-learn/scikit-learn/blob/844b4be24d20fc42cc13b957374c718956a0db39/sklearn/decomposition/_lda.py#L729
# <br>主题困惑度1：https://blog.csdn.net/weixin_43343486/article/details/109255165
# <br>主题困惑度2：https://blog.csdn.net/weixin_39676021/article/details/112187210

# ## 1.预处理

# In[3]:


import os
import pandas as pd
import re
import jieba
import jieba.posseg as psg
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
import numpy as np
import pyLDAvis.sklearn
import matplotlib.pyplot as plt


output_path = './save'
file_path = './data'
data = pd.read_excel("./data/data.xlsx")  # content type
dic_file = "./data/dict.txt"
stop_file = "./data/stopwords.txt"


# 将中文进行去stop_word的分词，返回 " ".join(seg_words)
def chinese_word_cut(mytext):
    jieba.load_userdict(dic_file)
    jieba.initialize()
    try:
        stopword_list = open(stop_file, encoding='utf-8')
    except:
        stopword_list = []
        print("error in stop_file")
    stop_list = []
    flag_list = ['n', 'nz', 'vn']
    for line in stopword_list:
        line = re.sub(u'\n|\\r', '', line)
        stop_list.append(line)

    word_list = []
    # jieba分词
    seg_list = psg.cut(mytext)
    for seg_word in seg_list:
        word = re.sub(u'[^\u4e00-\u9fa5]', '', seg_word.word)
        # word = seg_word.word  #如果想要分析英语文本，注释这行代码，启动下行代码
        find = 0
        for stop_word in stop_list:
            if stop_word == word or len(word) < 2:  # this word is stopword
                find = 1
                break
        if find == 0 and seg_word.flag in flag_list:
            word_list.append(word)
    return (" ").join(word_list)


# 对文档进行分词
data["content_cutted"] = data.content.apply(chinese_word_cut)
print(type(data), data.shape)


# 降序输出topic-doc DA 中 top-n概率的feature_words
def print_top_words(model, feature_names, n_top_words):
    tword = []
    for topic_idx, topic in enumerate(model.components_):
        print("Topic #%d:" % topic_idx)
        topic_w = " ".join([feature_names[i] for i in topic.argsort()[:-n_top_words - 1:-1]])
        tword.append(topic_w)
        print(topic_w)
    return tword


# 根据词频选取top-n features（seg_word）得到doc-words表 → [doc_num, n_features]
n_features = 1000  # 提取1000个特征词语
tf_vectorizer = CountVectorizer(strip_accents='unicode',
                                max_features=n_features,
                                stop_words='english',
                                max_df=0.5,     # max_document_frequence: 词出现的文档频率最大阈值
                                min_df=10)      # min_document_frequence，词出现的文档频数最小阈值
tf = tf_vectorizer.fit_transform(data.content_cutted)
print(type(tf), tf.shape)


# 基于词频特征矩阵获取doc-topic DA 和 topic-word DA → [doc_num, topic_num], [topic_num, n_features]
n_topics = 8
lda = LatentDirichletAllocation(n_components=n_topics,      # topic_num
                                max_iter=50,
                                learning_method='batch',
                                learning_offset=50,
                                # doc_topic_prior=0.1,      # doc-topic先验分布θ，缺省为1/n_components
                                # topic_word_prior=0.01,    # topic-word先验分布β，缺省为1/n_components
                                random_state=0)

lda.fit(tf)
print(lda.components_.shape)    # topic-word DA
topics = lda.transform(tf)      # topic-word DA
print(lda.doc_topic_prior_, lda.doc_topic_prior)
print(lda.topic_word_prior_, lda.topic_word_prior)

n_top_words = 25
tf_feature_names = tf_vectorizer.get_feature_names()
                                # 获取doc-words表中各feature对应的name
topic_word = print_top_words(lda, tf_feature_names, n_top_words)

topic = []
for t in topics:
    topic.append("Topic #" + str(list(t).index(np.max(t))))
data['概率最大的主题序号'] = topic
data['每个主题对应概率'] = list(topics)
data.to_excel("./save/data_topic.xlsx", index=False)


pic = pyLDAvis.sklearn.prepare(lda, tf, tf_vectorizer)
# pyLDAvis.display(pic)
pyLDAvis.save_html(pic, './save/lda_pass' + str(n_topics) + '.html')
# pyLDAvis.display(pic)
# 去工作路径下找保存好的html文件
# 和视频里讲的不一样，目前这个代码不需要手动中断运行，可以快速出结果


# ### 2.4困惑度

# LDA model hyper-parameter K evaluation test
plexs = []
scores = []
n_max_topics = 16
for i in range(1, n_max_topics):
    lda = LatentDirichletAllocation(n_components=i,
                                    max_iter=50,
                                    learning_method='batch',
                                    learning_offset=50, random_state=0)
    lda.fit(tf)
    plexs.append(lda.perplexity(tf))
    scores.append(lda.score(tf))


n_t = 15  # 区间最右侧的值。注意：不能大于n_max_topics
x = list(range(1, n_t + 1))
# plt.plot(x, plexs)
plt.plot(x, scores)
plt.xlabel("number of topics")
# plt.ylabel("perplexity")
plt.ylabel("score")
plt.show()
```