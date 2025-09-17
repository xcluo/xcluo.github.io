## Slimmable Neural Networks
> 论文：Slimmable Neural Networks  
> Github：[slimmable_networks
](https://github.com/JiahuiYu/slimmable_networks)  
> University of Illinois at Urbana-Champaign & Snap Inc. & ByteDance Inc, 2018 Dec, ICLR 2019


### 主要内容
- 训练时随机选取 n_head, d_model 或 d_ffn
- Attention: `(bs, n_head, seq_len, head_dim)`
- FFN: `(bs, seq_len, d_ff) → (bs, seq_len, d_model)`
- LN: `(bs, seq_len, d_model) 进行LN`
- 推理阶段设置 n_head, d_model 和 d_ffn
- 不像MRL一样对于单batch样本进行线性相加，由于LN（或RMSNorm）特性，只能单次计算选定width的loss