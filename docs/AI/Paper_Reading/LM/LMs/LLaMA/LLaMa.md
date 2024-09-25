#### 摘要
1. 参数405B，最长支持128K context window，dense(vanilla) rather than MoE transformer
2. 多语种、代码、推理能力全面增强，comparable with GPT-4
3. （保证自然语言能力前提下）增量提升语音，视频以及图像等模态处理功能
 

#### 预训练数据
数据量规模更大，质量更高，最终获得15T 截至日期为2023年底的多语种tokens, 与之对应的llama 2则使用了1.8T tokens

1. 数据来源  
    - 数据源版权不便展开细说，如common crawl、网络图书、github code、youtube字幕等

2. 数据清洗  
    - 过滤包含大量个人隐私信息以及成人内容数据
    - 去除网页广告、html tags（保留code以及推理内容等复杂tags）
    - 去重
        - url-level：保留各url的最新版网页数据
        - document-level：MinHash de-duplication
        - line-level：in 30M lines，#freq > 6 $\rightarrow$ remove。大多为导航菜单、cookie等文本，**去除后能有效提升预料质量**
    - heuristic filtering
        - n-gram coverage ratio > p $\rightarrow$ remove。大多为无意义数据，如日志或报错数据（日志数据通常唯一，无法通过简单uniq去重）
        - dirty word count，黑名单关键词过滤
        - 比较爬取语料和既有训练数据集KL散度，数值超过一定范围的视为予以删除
    - model-based filtering
        - fasttext, RoBERTa去除低质量内容
3. 确定预训练数据  
    尽可能地使用高质量的数据参与训练
    - 知识分类：将语料进行分类，并对模型效果贡献不大的类别（对艺术和娱乐板块预料）进行较少的采样
    - data mix：~50%通用数据，25%数学和推理数据，17%代码数据，8%多语种数据


4. annealing data
    模型预训练末期（非SFT），使用少量但质量特别高的数据进行收尾（如高质量的code和数学数据）

#### 预训练
1. 模型结构  
      - llama 3.1，16k H100 (405B), compared to llama 2 using A100
      - GQA(grouped query attention) with 8 key-value heads
      - 同一个sequence中，隔绝不同文档间的数据attention（对模型效果提升不大，但是对训练超长context window影响较大）
      - vocabulary with 128k tokens，100k from tiktoken and 28k additional non-English tokens
      - BPE $\rightarrow$ tiktoken（BPE的高效优化实现），后者具有更好的压缩比3.17 char per token $\rightarrow$ 3.94 char per token
      - 模型效果提升主要还是数据质量、多样性和规模导致的
    <div class="one-image-container">
    <img src="\AI\Paper_Reading\LM\LMs\LLaMA\image\llama3_hyperparameters.jpg" style="width: 80%;">
    <p style="text-align: center;">llama 3模型主要超参</p>
    </div>
1. pre-training
      - next token predication，力大砖飞
      - 大力出奇迹且简单：sft + RS(reject sampling) + DPO，效果提升主要还是数据质量高和多样性
      - 70b is comparable better
2. scaling law  
    单纯的增大模型规模或增加训练样本可能会导致undertrained，导致算力或语料库未充分利用
    - 预测最佳模型规模：统计各种算力情况下，模型在不同规模数据集下预训练后，在下游任务benchmark validation set效果表现的相关性（之前仅基于预训练的next token predication loss预测的方法太过于粗糙了，噪声较大且准确性较低）
    - assume $\#tokens=AC^\alpha$，A为标量，C表示算力支持
    <div class="one-image-container">
    <img src="\AI\Paper_Reading\LM\LMs\LLaMA\image\llama3_scale_law_model-size.jpg" style="width: 100%;">
    <p style="text-align: center;">llama 3 scale law. (fixed  compute=#tokens*model_size, #tokens $\uparrow$，model_size $\downarrow$)</p>
    </div>
    - 预测最佳模型规模下效果：通过（最佳模型规模）小模型和既有的llama2模型在ARC benchmark上效果，预测相应规模下llama 3的效果
    <div class="one-image-container">
    <img src="\AI\Paper_Reading\LM\LMs\LLaMA\image\llama3_scale_law_accuracy.jpg" style="width: 100%;">
    </div>
3. large scale pre-training strategy
    - Parallelism：PP(pipeline parallelism), TP(tenspr parallelism), DP(data parallelism), CP(context parallelism)
    
4. pre-training recipe
    预训练分为三个主要步骤，即
    - initial pre-training
          - (0, 252M] tokens, seq_len=4k, batch_size=4M/seq_len
          - (252M, 2.87T] tokens, seq_len=8k, batch_size=8M/seq_len
          - after 2.87T tokens, seq_len=8k, batch_size=16M/seq_len
    - long-context (128k) pre-training
        - final stage of pre-training, 0.8T tokens(0.8/15=5.33%占比) for long-context pre-training
        - 6个step逐次增加seq_len，直到模型能够适应增长后的seq_len（1. 短文本任务效果保持不变；2. 在当前context window很好地解决大海捞针问题）
    - annealing
        - 训练末期通过超高质量数据40M tokens（lr decay→0，seq_len=128k）来收尾模型，output=avg(multi finals checkpoints)





-----------------
- Guard 3可以多prompt的输入输出进行一些安全上的改写
preference data
- 使用多个模型对给定prompt进行生成，并采样两条样本（由不同模型生成）