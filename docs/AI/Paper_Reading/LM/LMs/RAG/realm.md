## REALM
> 论文：REALM: **RE**trieval-**A**ugmented **L**anguage **M**odel Pre-Training  
> Google Research 2020 Feb, ICML 2020  


### 主要内容
- using sentence x with some tokens (z) masked out, retrieve potential documents z, and then extract (not generate, 因此先验地假设目标y为出现在z中的连续span) y from z
- $p(y|x) = \sum_{z \in Z} p(z|x)*p(y|x, z)$
- performance-base signal from unsupervised text: 奖励提升perplexity的文档分数，惩罚削弱perplexity的文档分数