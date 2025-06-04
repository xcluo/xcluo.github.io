#### Label Smoothing
标签平滑是一种用于分类任务的正则化技术，旨在缓解模型对训练标签的过度自信（over-confidence）问题，==将硬标签替换为软标签==，防止过拟合，提升模型的泛化能力。

1. 真实类别的概率为 $1-\epsilon$，其中 $\epsilon$ 为平滑系数  
2. 其它类别的概率均匀分配 $\epsilon / (K-1)$，其中 $K$ 为类别数  


!!! info ""
    - 也可将平滑系数$\epsilon$均匀分配给所有类别，即 $y_\text{LS}=(1-\epsilon)\cdot y + \epsilon/K$