## GAE
> 论文：High-Dimensional Continuous Control Using **G**eneralized **A**dvantage **E**stimation  
> University of California Berkeley 2015 Jun, ICLR 2016

### 主要内容

- reduce bias：$\hat{V}(s_t) = r_t + \gamma V(s_{t+1})$
- reduce variance：$\hat{V}(s_t) = r_t + \gamma r_{t+1} + \dots + \gamma^{T-t+1}r_{T-1} + \gamma^{T-t}V_{s_t}$
- policy gradient estimators that significantly reduce variance while maintaining
a tolerable level of bias  
- 通过衰减参数 $\gamma$ 降低与延迟后续步骤的相应奖励来减少方差，但代价是引入了偏差（衰减后值与原真实奖励值存在差异）
- 有折扣优势函数$A^{\pi, \gamma}$，无折扣优势函数$A^{\pi}$
- trust region optimization for the value function  
- bias-variance tradeoff  


```
We’ve described an advantage estimator with two separate parameters and , both of which contribute to the bias-variance tradeoff when using an approximate value function. However, they serve different purposes and work best with different ranges of values. most importantly determines the scale of the value function V ; , which does not depend on . Taking  < 1 introduces bias into the policy gradient estimate, regardless of the value function’s accuracy. On the other hand,  < 1 introduces bias only when the value function is inaccurate. Empirically, we find that the best value of  is much lower than the best value of , likely because  introduces far less bias than  for a reasonably accurate value function.
```