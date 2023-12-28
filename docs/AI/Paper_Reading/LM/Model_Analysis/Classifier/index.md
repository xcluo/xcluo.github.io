机构：

- 北京大学CS、WeChat AI

论文地址：  

- [LabelWords are Anchors: An Information Flow Perspective for Understanding In-Context Learning]() (2023 EMNLP Best Paper)

Github地址：  

- [https://github.com/lancopku/label-words-are-anchors](https://github.com/lancopku/label-words-are-anchors)

---
### Abstarct
文章先假设LM模型（AR模型）中样本标签词作为信息锚点（Anchor）并验证结果，即：在浅层中样本标签词能够聚集提取、收集语义信息，并在深层将提取的语义汇聚、流向模型最终预测的分类类别。

<div class='one-image-container'>
    <img src='\AI\Paper_Reading\LM\Model_Analysis\Classifier\information_flow.png' width='90%'>
    <p>ICL信息流示意图</p>
</div>

### Result
1. 实验发现样本标签词起到信息锚点的作用  
      - 在浅层中样本标签词能够聚集提取、收集语义信息
      - 在深层标签词将提取的语义汇聚、流向模型最终预测的分类类别。
2. 基于标签词起到信息锚点作用的现象，对于分类任务设计并实现  
      - anchor re-weighting method，进一步提升ICL效果
      - AR模型输入压缩，加快inference性能

### Limitation
1. 暂时只局限于分类任务暂时无法泛化至生成任务
2. 目前只在传统ICL模式上进行试验，其他ICL（如chaio-of-thought prompt链式思维）暂未探索
3. 目前只在GPT-XL和GPT-J模型上实现，更大规模的LLM模型效果未知

### Motivation
ICL（In-Context Learning）在训练LM时效果明显，但是针对其为何奏效的工作暂未被较好地研究探明。提出了以下标签词作为信息锚点的假设，具体为：

- $\mathcal{H_1}$：在浅层中，标签词收集演示样本信息以形成更深层的语义表示。  
- $\mathcal{H_2}$：在深层中，模型从标签词中提取信息以形成最终的预测。

### Details
#### Prompt Construction
<div class='one-image-container'>
    <img src='\AI\Paper_Reading\LM\Model_Analysis\Classifier\prompt_construction.png'>
    <p>prompt template=(Num*C) demonstrations + input_to_be_predicted。<br>Num表示每个类别样本数，一般为1；C表示类别数；demonstration表示相应类别的随机样本；</p>
</div>

#### 指标
1. 显著性分数（Attention矩阵的一阶泰勒展开）

    $$I_l=\sum_h\lvert A^T_{h,l}\frac{\partial \mathcal{L}(x)}{\partial A_{h,l}} \rvert$$
    > $l$ 表示Transformer 具体的层  
    > $h$ 表示 `attention` 的 `#heads`   
    > $I_l(i, j)$ 表示 $l\text{-th}$ 层中位置 $j$ 至位置 $i$ 的信息流重要性

1. 文本内容至标签词信息流重要性均值 $S_{wp}$

    $$
    \begin{aligned}
        & S_{wp}=\frac{\sum_{(i, j) \in C_{wp}} I_l{(i, j)}}{\lvert C_{wp} \rvert} \\
        & C_{wp} = \{(p_k, j): k \in [1, C], j < p_k\}
    \end{aligned}
    $$

2. 标签词至分类标签信息流重要性均值 $S_{pq}$
 
    $$
    \begin{aligned}
        & S_{pq} = \frac{\sum_{(i, j) \in C_{pq}} I_l{(i, j)}}{\lvert C_{pq} \rvert} \\
        & C_{pq}= \{(q, p_k): k \in [1, C]\} 
    \end{aligned}
    $$

3. 除去上述两种外所有词信息流重要性均值 $S_{ww}$

    $$
    \begin{aligned}
        & S_{ww} = \frac{\sum_{(i, j) \in C_{ww}} I_l{(i, j)}}{\lvert C_{ww} \rvert} \\
        & C_{ww} = \{(i, j):j \lt i\} - C_{wp} - C_{pq}
    \end{aligned}
    $$

#### 验证假设H1、H2

<div class='row-image-container'>
    <img src='\AI\Paper_Reading\LM\Model_Analysis\Classifier\three_s_scores_on_sst-2.png'>
    <img src='\AI\Paper_Reading\LM\Model_Analysis\Classifier\three_s_scores_on_agnews.png'>
</div>
>- $S_{wp}$ 在浅层重要性高，$S_{pq}$ 在深层重要性高
>- $S_{wp}$ 和 $S_{pq}$ 重要性都比 $S_{ww}$ 高


#### 进一步验证假设H1
<div class='one-image-container'>
    <img src="\AI\Paper_Reading\LM\Model_Analysis\Classifier\validate_h1_royality.png" width='60%'>
    <p>截断前/后5层信息流前后效果前后对比</p>
</div>
>- **isolate label words**: $A_l(p, i)=0, i\lt p$
>- **isolate random non-label**: 截断#C个非标签词间的信息流，类似于MASK
>- **label loyality**: 前后输出标签一致性
>- **word loyality**: 前后输出top-5的Jaccard similarity $J(A, B)=\lvert A \cap B \rvert / \lvert A \cup B \rvert$

<div class="row-image-container">
    <div>
        <img src="\AI\Paper_Reading\LM\Model_Analysis\Classifier\validate_h1_extend_on_gptxl.png">
    </div>
    <div>
        <img src="\AI\Paper_Reading\LM\Model_Analysis\Classifier\validate_h1_extend_on_gptj.png">
    </div>
</div>

>- 消融实验表明截断浅层信息流模型效果有明显影响，且在随着截断层数增加，效果持续下降
>- 消融实验表明截断深层信息流模型效果影响不大，即使截断的层数相对较多
>- 截断标签词的信息提取流比非标签词的信息流影响更大

#### 进一步验证假设H2
**动机**：发现现象并提出先验性假设 $A_L=\sum_h {A_{hl}}$ 中的标签词位置至分类类别位置处的信息流与输出标签强相关，即 $A_l(q, p_1), ..., A_l(q, p_C) \backsim p_{f1}, ..., p_{fC}$ 。

<div class="row-image-container">
    <div>
        <img src="\AI\Paper_Reading\LM\Model_Analysis\Classifier\validate_h2_on_gptxl.png">
        <p>GPT-XL上的效果表现：$\text{AUCORC}_l$ 和 $R_l$ </p>
    </div>
    <div>
        <img src="\AI\Paper_Reading\LM\Model_Analysis\Classifier\validate_h2_on_gptj.png">
        <p>GPT-J上的效果表现：$\text{AUCORC}_l$ 和 $R_l$ </p>
    </div>
</div>
>- $\text{AUCORC}_l$：$l\text{-th}$ 层 $A_l(q, p_i)$ 与输出标签的相关性的ROC值
>- $R_l=\frac{\sum_{i=1}^{l}(\text{AUCROC}_i - 0.5)}{\sum_{i=1}^{N}(\text{AUCROC}_i - 0.5)}$ 表示前 $l$ 层对模型输出影响占比

**结论**：

- 浅层对模型输出影响占比较小
- 随着层数增加（>middle），$R_l$增大较为明显，表现为深层对模型输出影响较大
- **验证结论**：$A_L=\sum_h {A_{hl}}$ 中的标签词位置至分类类别位置处的信息流与输出标签强相关，即 $A_l(q, p_1), ..., A_l(q, p_C) \backsim p_{f1}, ..., p_{fC}$ 

### Application
基于结论，<span style='color: blue'>$A_L=\sum_h {A_{hl}}$ 中的标签词位置至分类类别位置处的信息流与输出标签强相关，即 $A_l(q, p_1), ..., A_l(q, p_C) \backsim p_{f1}, ..., p_{fC}$ </span>，进行了以下深入研究

#### Anchor Reweighting Method
**动机**：

$$
\begin{aligned}
\text{P}_f(Y=i|X=x) &\approx A(q, p_i) \\
 &=\frac{\text{exp}(q_qk_{p_i}^T/\sqrt{d})}{\sum_{j=1}^{seq\_len}\text{exp}(q_qk_{j}^T/\sqrt{d})}  \\
\end{aligned}
$$

$$
\begin{aligned}
 \log \frac{\text{P}_f(Y=i|X=x)}{\text{P}_f(Y=C|X=x)} &= (k_{p_i} - k_{p_C})^Tq_q/\sqrt{d} \\
 &\Rightarrow \beta_0^i + (k_{p_i} - k_{p_C})^Tq_q/\sqrt{d} \\
 \hat{A}(q, p_i)&=\exp(\beta_0^i)A(q, p_i)
\end{aligned}
$$
> Attention Variant：对于每层每个head，设计一个learnable参数 $\beta_{hl} \in \mathbb{R}^C$

除cross-entropy外，新增以下辅助损失函数来线性调整各样本标签重要性

$$
\beta^* = \arg \min_{\beta}\mathcal{L}(X_{train}, Y_{train})
$$

<div class='one-image-container'>
    <img src='\AI\Paper_Reading\LM\Model_Analysis\Classifier\anchor_reweight_result.png'>
</div>
> few-shot场景下效果优化明显

#### Demonstrations Compression
AR模型是单向的，对于Prompt：<span style='color: blue'>(Num*C) demonstrations</span> + input_to_be_predicted的前半部分 其实是固定的，没有必要重复计算，处于性能考量，进行了压缩。实验对比包括

- <span style='font-weight: bold'>$\text{Hidden}_{anchor}$</span>：<span class=underline_span>不需要(Num*C) demonstrations，在每层的hidden state前连接已计算好的C个 "formating + label" 的hidden state</span>，其中formating（如SST-2中的Sentiment:）信息能够帮助模型确定输出空间
- <span style='font-weight: bold'>$\text{Text}_{anchor}$</span>：<span class=underline_span>将(Num*C) demonstrations改为对应的C个"formating + label"</span>
- <span style='font-weight: bold'>$\text{Hidden}_{random}$</span>：与<span class=underline_span><span style='font-weight: bold'>$\text{Hidden}_{anchor}$</span> 的区别在于该方案连接的是C个"formatting + random non-label word"</span>
- <span style='font-weight: bold'>$\text{Hidden}_{random-top}$</span>：随机了20个<span style='font-weight: bold'>$\text{Hidden}_{random}$</span>并取了最好的结果

<div class='one-image-container'>
    <img src='\AI\Paper_Reading\LM\Model_Analysis\Classifier\demonstration_compression_result.png' width="60%">
</div>

<div class='one-image-container'>
    <img src='\AI\Paper_Reading\LM\Model_Analysis\Classifier\demonstration_compression_speedup_on_lm.png' width="56%">
    <img src='\AI\Paper_Reading\LM\Model_Analysis\Classifier\demonstration_compression_speedup_on_length.png' width="42%">
</div>


>- 进一步确证标签词提取了重要的语义信息                     
>- 模型越大，加速效果越明显；被压缩的文本长度越长，加速效果越明显
>- demonstartion compression方法能够提升模型性能且效果下降影响不大
