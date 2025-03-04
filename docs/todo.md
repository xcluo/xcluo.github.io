- [ ] MinHash de-duplication
- Frobenius范数，次可加性$\Vert A+B \Vert_{F}\le \Vert A \Vert_F + \Vert B \Vert_F$，空间向量相加，两边之和大于第三边
- [ ] sft roberta with multiple sequence concurrent with customized attention mask
- [ ] DeepSeek
- [ ] instruct gpt: Training language models to follow instructions with human feedback
- [ ] [大模型面试](https://zhuanlan.zhihu.com/p/691588703)
- [ ] NSA: natively trainable sparse Attention
- [ ] Wizardmath: Empowering mathematical reasoning for large language models via reinforced evol-instruct
- [ ] Math-shepherd: Verify and reinforce llms step-by-step without human annotations
- [ ] vllm: Efficient Memory Management for Large Language Model Serving with PagedAttention
- [ ] ollama, tensorRT
- [ ] hallucination
- [ ] 模型DP时，多个数据loss结果会进行交互all reduce
- [ ] [gradient checkpointing](https://www.bilibili.com/video/BV1nJ4m1M7Qw/?spm_id_from=333.1387.search.video_card.click&vd_source=782e4c31fc5e63b7cb705fa371eeeb78): Training Deep Nets with Sublinear Memory Cost
- [ ] accelerate config
- [ ] label smoothing
- [ ] importance sampling
- [ ] DPO, PPO, GRPO
- [ ] Prefix Tuning， Prompt Tuning
- [ ] 模型幻觉
- [ ] DeepSpeed（Zero Redundancy Optimizer）、Megatron-LM、HAI-LLM framework（higher flyer）
- [x] Pre-Norm与Post-Norm的区别与选择
- [x] KV cache：将L层K与V进行缓存以执行Attention
- [x] Decouped RoPE
- [x] LDA潜在迪利克雷分布，b站视频 LDA主题模型
- [ ] LSA/PLSA
- [ ] Cholesky分解
- [ ] [odds，logit，ligitis](https://zhuanlan.zhihu.com/p/435912211)
- [ ] [GBDT + LR](https://www.cnblogs.com/wkang/p/9657032.html)
- [ ] Restricted Boltzmann Machines (RBM)
- [ ] A/B test
- [ ] TF-IDF_j, MI_{a, b, c, d}
- [x] evaluation metrics: MRR, HR, NDCG
- [x] Attention softmax后除以$\sqrt{d_k}$是因为权重矩阵中每个元素都是通过两个(1， d_k)方差为1的向量相乘得到的，基于正态分布累加后的标准差公式可知该值方差变为$\sqrt{d_k}$，因此执行该操作，不除以$\sqrt{d_k}$，根据softmax函数曲线，softmax结果表现更倾向于one-hot分布，[会带来梯度消失问题](https://spaces.ac.cn/archives/8620/comment-page-4#comment-24076)
- tensorflow 1.x中在梯度下降时如何设置L1，L2正则化约束
    ```python
    with tf.variable_scope("layer1", regularizer=l2_regularizer):
        xxx
    reg_losses = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
    total_loss = cross_entropy_loss + sum(reg_losses)
    ```
- 推荐系统攻击
- [推荐系统论文笔记](https://github.com/Doragd/Algorithm-Practice-in-Industry/blob/main/%E6%90%9C%E5%B9%BF%E6%8E%A8%E7%AE%97%E6%B3%95%E7%B3%BB%E5%88%97%E4%B8%B2%E8%AE%B2.md#%E6%8E%A8%E8%8D%90%E7%B3%BB%E7%BB%9F%E8%AE%BA%E6%96%87%E7%AC%94%E8%AE%B0)
    - [x] 推荐系统论文精读
    - [x] 经典推荐算法学习
    - [x] 推荐系统与深度学习论文笔记
- https://km.netease.com/v4/detail/blog/223053  
- https://readpaper.feishu.cn/docx/CrMGdSVPKow5d1x1XQMcJioRnQe
- Gradient Checkpointing，[gif](https://pic3.zhimg.com/v2-1679b74a85687cdb250e532931bb266a_b.webp)
