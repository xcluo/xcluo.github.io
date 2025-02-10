### MQA
> 论文：MQA: Fast Transformer Decoding: One Write-Head is All
You Need  
> **MQA**：**M**ulti-**Q**uery **A**ttention  
> Google Noam Shazeer, 2019


#### 工作要点
1. 所有的heads使用同一个K和V，即【$h\ \text{Q} + 1\ \text{K} + 1\ \text{V}$】
2. 是一个性能与效果的trade-off，通过牺牲一点点效果表现，提升速度的同时减少GPU使用量
    <div class="one-image-container">
        <img src="\AI\Paper_Reading\Trick\Memory_Saving\Attention_Variants\image\mqa_time_performance.png" style="width: 95%;">
        <!-- <p style="text-align: center;">图片标题</p> -->
    </div>
3. 加速原因
    - [x] 少了$h-1$次K、V结果计算；
    - [x] K、V缓存减少，为更高速SRAM腾出了更多空间用以加速计算。


### GQA
> 论文：GQA: training generalized multi-query transformer models from multi-head checkpoints  
> **GQA**：**G**rouped-**Q**uery **A**ttention  
> Google Research, EMNLP 2023

#### 工作要点
1. $\frac{h}{g}$个heads使用同一个K和V，即【$GQA_g=h\ \text{Q} + g\ \text{K} + g\ \text{V}$】

    - 通过控制GQA的超参$g$，可得到特例MHA和MQA，前者$g=h$，后者$g=1$
    - 如果要兼容MHA或MQA，可以通过各组K,V求平均或复制的方式快速实现
    <div class="one-image-container">
    <img src="\AI\Paper_Reading\Trick\Memory_Saving\Attention_Variants\image\mha_mqa_gqa_diagram.png" style="width: 90%;">
    <!-- <p style="text-align: center;">图片标题</p> -->
    </div>

2. 是一个性能与效果的trade-off，通过牺牲一点点效果表现，提升速度的同时减少GPU使用量
3. 加速原因：MHA与MQA的trade-off产物


### MLA
> 论文：DeepSeek-V2: A Strong, Economical, and Efficient Mixture-of-Experts Language Model  
> MLA: **M**ulti-head **L**atent **A**ttention  
> DeepSeek-AI, 2024