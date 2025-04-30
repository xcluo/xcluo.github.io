## GAE
> 论文：High-Dimensional Continuous Control Using **G**eneralized **A**dvantage **E**stimation  
> University of California Berkeley 2015 Jun, ICLR 2016



- policy gradient estimators that significantly reduce variance while maintaining
a tolerable level of bias  
- 通过衰减参数 $\gamma$ 降低与延迟后续步骤的相应奖励来减少方差，但代价是引入了偏差（衰减后值与原真实奖励值存在差异）
- 有折扣优势函数$A^{\pi, \gamma}$，无折扣优势函数$A^{\pi}$
- trust region optimization for the value function  
- bias-variance tradeoff  
- 