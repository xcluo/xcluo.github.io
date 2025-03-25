## REALM
> 论文：REALM: **RE**trieval-**A**ugmented **L**anguage **M**odel Pre-Training  
> Google Research 2020 Feb, ICML 2020  


### 主要内容
- $p(y|x) = \sum_{z \in Z} p_{\theta}(z|x)*p_{\phi}(y|x, z)$，Z为top-k的候选文档
- 后计算的MIPS获取top-k，
- 预训练阶段使用stale optimization
- 基于上述公式可知检索阶段是抽象潜在的，没有训练目标进行控制，因此需要通过鼓励机制提升该阶段能力
- performance-base signal from unsupervised text: 奖励提升perplexity的文档分数，惩罚削弱perplexity的文档分数
- using sentence x with some tokens (z) masked out, retrieve potential documents z, and then extract (not generate, 因此先验地假设目标y为出现在z中的连续span) y from z


