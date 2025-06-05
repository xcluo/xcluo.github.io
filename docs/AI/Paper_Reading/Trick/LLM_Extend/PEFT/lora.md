### LoRA
> 论文：**Lo**w-**R**ank **A**daptation of large language models  
> Github: [LoRA-GA: Low-Rank Adaptation with Gradient Approximation](https://github.com/microsoft/LoRA)  
> MicroSoft, ICLR, 2022

#### 工作要点
1. lora先验地假设可通过lora思想将分解的低秩矩阵来近似待调整的高维矩阵
2. 冻结预训练模型的参数，在通过额外插入trainable rank decomposition matrices达到微调目的
3. on GPT-3 175B, trainable参数量减少为10000分之一，GPU使用减少为3分之一
4. 效果和直接训练差不多，且infer性能消耗区别不大


#### 细节实现  
$W=W_o + \frac{\alpha}{r} BA, where\ W, W_o \in \mathbb{R}^{d, k}, A \in \mathbb{R}^{d, r}, B \in \mathbb{R}^{r, k}, r\ll\min(d, k)$

<div class="one-image-container">
    <img src="\AI\Paper_Reading\Trick\LLM_Extend\LLM_SFT\image\lora_diagram.jpg" style="width: 30%;">
    <!-- <p>LoRA在Attention各部分权重上的消融实验效果</p> -->
</div>

- 超参 $r$，一般1, 2, 4, 8；$r\ll\min(d, k)$，而且**一味地增大$r$效果提升有限**，如果下游任务数据集和预训练语料库差别较大可以适当增大。
<div class="one-image-container">
    <img src="\AI\Paper_Reading\Trick\LLM_Extend\LLM_SFT\image\lora_vary_rank.jpg" style="width: 90%; " >
    <!-- <p>LoRA在Attention各部分权重上的消融实验效果</p> -->
</div>
- $\alpha$为标量（一般$\alpha\ge 2*r$），值越大LoRA模块占比重越高，反之同理。
- $A$ is initialized by $\mathcal{N}(0, \sigma ^2)$
- $B$ is initialized by 0

#### LoRA方案选择  
- 消融实验效果表明单对$W_q$或$W_k$应用LoRA效果较差，**较好的lora方案是【$W_q$ + $W_v$】**
<div class="one-image-container">
    <img src="\AI\Paper_Reading\Trick\LLM_Extend\LLM_SFT\image\lora_weight_type_ablation.jpg" style="width: 80%;">
    <p>LoRA在Attention各部分权重上的消融实验效果</p>
</div>


#### 性能影响
- 当`batch_size`较大时，引入的LoRA模块对模型infer性能影响可以忽略不计；
- 当`batch_size`较小时，引入的LoRA模块对模型infer性能存在影响，且最高达30%。
<div class="one-image-container">
    <img src="\AI\Paper_Reading\Trick\LLM_Extend\LLM_SFT\image\lora_performance_comparison.jpg" style="width: 100%;">
    <!-- <p>LoRA在Attention各部分权重上的消融实验效果</p> -->
</div>
