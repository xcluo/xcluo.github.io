### FGSM
> 论文：Explaining and Harnessing Adversarial Examples  
> FGSM：**F**ast **G**radient **S**ign **M**ethod  
> Google, ICLA 2015

#### 工作要点
- [x] 在梯度方向通过控制超参$\epsilon$增加些微扰动（特别是梯度上升方向）即可轻易使模型严重误判，
    
    $$
    \begin{aligned}
    f&(x+r)\ne f(x) \\
    r& = \epsilon \text{sign}\big(\nabla \mathcal{L}_x(\theta, x, y)\big)
    \end{aligned}
    $$

    !!! info ""
        类似于对梯度附加了一个$L_\infty$ norm约束（标准$L_\infty$除以最大绝对值，此处额外增加了取整步骤）
        
- [x] 对抗训练提升模型泛化能力$\mathcal{L}(\theta, x, y)=\mathcal{L}_1(\theta, x, y) + \alpha\mathcal{L}_1\big(\theta, x+\epsilon\text{sign}(\nabla_x \mathcal{L}_1(\theta, x, y)\big), y)$

    !!! info ""    
        $\alpha$一般取1
    
- 【梯度上升方向（+noise）降低置信、梯度下降方向（-noise）提升置信】，属于白盒攻击，还可以对梯度进一步应用l2 norm，即$\epsilon\frac{g}{||g||_2}$，https://jaketae.github.io/study/fgsm/
- 可以理解为l_∞约束，$\frac{g}{||g||_\infty}$，只不过权值不是归一化而是进行sign变换
- 通过对【对抗样本 + 干净样本】的数据混合训练，神经网络能实现一定程度上的正则化与泛化增强，即下式

#### 主要内容
1. FGSM不同$\epsilon$下的效果表现  
    <div class="row-image-container">
        <div>
            <img src="\AI\Paper_Reading\Trick\Denoising\AdversarialTraining\image\fgsm_-0.1g.jpg" style="width: 99%">
            <p>-$0.01\nabla_x$: 0.98拉多</p>
        </div>
        <div>
            <img src="\AI\Paper_Reading\Trick\Denoising\AdversarialTraining\image\fgsm_0g.jpg" style="width: 99%">
            <p>$0\nabla_x$: 0.42拉多</p>
        </div>
        <div>
            <img src="\AI\Paper_Reading\Trick\Denoising\AdversarialTraining\image\fgsm_0.01g.jpg" style="width: 99%">
            <p>+$0.01\nabla_x$: 0.13沙克犬</p>
        </div>
        <div>
            <img src="\AI\Paper_Reading\Trick\Denoising\AdversarialTraining\image\fgsm_0.1g.jpg" style="width: 99%">
            <p>+$0.1\nabla_x$: 0.15威玛猎犬</p>
        </div>
    </div>
    <p style="text-align:center">在图像梯度(上升/下降)方向施加不同程度扰动对模型识别结果的影响</p>

    !!! info ""
        在梯度上升方向增加扰动，模型置信降低，反之增加；

2. FGSM对抗训练提升模型稳定性  
    $\mathcal{L}(x, y, \theta)=\mathcal{L}_1(x, y, \theta) + \alpha\mathcal{L}_1\big(x+\epsilon\text{sign}(\nabla_x \mathcal{L}_1(\theta, x, y)\big), y, \theta)$


### approximation of LDS
> LDS：**L**ocal **D**istributional **S**moothness  
> 论文：Distributional Smoothing with Virtual Adversarial Training  
> Kyoto University, ICLA 2016   

</bar>

> 论文：Virtual Adversarial Training: A Regularization Method for Supervised and Semi-Supervised Learning  
> Preferred Networks & Kyoto University & ATR Cognitive Mechanisms Laboratories & Ritsumeikan University, TPAMI 2017

#### 工作要点
- [x] 虚拟对抗训练的半监督算法(VAT, Vitural Adversarial Training)，运用了平滑思想旨在使模型对处于一定的区间范围内的数据样本都有较为相似的分类结果。

    $$
    \begin{aligned}
        \mathcal{L} &= \mathop{\mathcal{L}_1}\limits_{(x_1,y_1) \in D_{label}}(x_1, y_1, \theta) + \alpha \mathop{\text{KL}}\limits_{x_2 \in D_{unlabel}}[p(Y|x_2,\theta)||p(Y|x_2+r_{\text{v-adv}},\theta)] \\
        r_{\text{v-adv-}2} &= \epsilon\frac{g}{||g||_2}, \text{where}\ g=\nabla_{x_2}\text{KL}[p(y|x_2,\theta)||p(y|x_2+r,\theta)]\Big\vert_{r=\xi d} \\
        r_{\text{v-adv-}\infty} &= \epsilon \text{sign}(g), \text{where}\ g=\nabla_{x_2}\text{KL}[p(y|x_2,\theta)||p(y|x_2+r,\theta)]\Big\vert_{r=\xi d}
    \end{aligned}
    $$

    !!! info ""
        - 局部平滑目标，即扰动范围$||r||_{2/\infty}\le \epsilon$内，$f(x)\approx f(x+r)$  
        - 两个超参 $\epsilon$ 和 $\alpha$，以及指定norm方式  
        - $(x_1,y_1)\in D_{label}, x_2\in D_{unlabel}$，前者有监督训练，后者无监督局部分布平滑，因此为半监督训练
        - 单次训练需要3次前向计算：`forward(x), forward(x2_update_r), forward(x2_get_final_kl)`
        - 单次后向需要2次后向计算：`backward(x2_update_r), backward(update_θ)`

- [x] VAT较（使用伪标签的）对抗训练模型的泛化能力优秀（AT新增训练点，VAT直接泛化面）

#### 主要内容
1. approximation of LDS

    $$
    \begin{aligned}
        \text{LDS}(r, x, \theta) =& \text{KL}(r, x, \theta)=\text{KL}[p(y|x,\theta)||p(y|x+r,\theta)] \\
        \approx& \text{LDS}(0, x, \theta) + \nabla_r\text{LDS}(0, x, \theta)|_{r=0} + \frac{1}{2}r^TH(x,\theta)r \\
        =& \frac{1}{2}r^TH(x,\theta)r \\
        r_{\text{v-adv}} =& \mathop{\text{arg max}}\limits_{r;\ ||r||_{2/\infty}\le \epsilon} \text{KL}(r, x, \theta) \\
        \approx& \epsilon *\overline{d(x, \theta)} \\
        \text{power iteration method} \\
        d\leftarrow& \overline{Hd} \\
        \text{finite differce method} \\
        Hd \approx& \frac{\nabla_r\text{KL}(0, x+\epsilon d,\theta)\big\vert_{r=0} - \nabla_r\text{KL}(0, x,\theta)\big\vert_{r=0}}{\epsilon d}*d \\
        =& \frac{\nabla_r\text{KL}(0, x+\epsilon d,\theta)\big\vert_{r=0}}{\epsilon} \\
        d=&\overline{\nabla_r\text{KL}(0, x+\epsilon d,\theta)\big\vert_{r=0}}
    \end{aligned}
    $$

    !!! info ""
        - $r=0$ 时KL散度值为0且对应极值点，所以泰勒展开式前两项均为0    
        - $\overline{\cdot}$ 为 L2 norm 操作，$u(x,\theta)$表示为海森矩阵$H$的最主要特征向量  
        - 特征值幂迭代近似算法  
            1. random_init $d$; 
            2. 迭代计算K次（此处1次效果就很理想）: $d = L_2\text{-}norm(Hd)$; 
            3. return $d$

2. K=1即可取得理想效果
    <div class="one-image-container">
        <img src="\AI\Paper_Reading\Trick\Denoising\AdversarialTraining\image\approximation_lds_vary-k.jpg" style="width: 50%;">
        <!-- <p style="text-align: center;"></p> -->
    </div>


3. VAT训练模型的泛化能力效果更强（AT新增训练点，VAT直接泛化面）
   <div class="one-image-container">
        <img src="\AI\Paper_Reading\Trick\Denoising\AdversarialTraining\image\approximation_lds_vat_process.jpg" style="width: 90%;">
        <p style="text-align: center;">VAT训练模型的泛化能力效果演化</p>
    </div>

### FGM
> 论文：Adversarial Training Methods for Semi-supervised Text Classification  
> FGM：**F**ast **G**radient **M**ethod  
> Github：[adversarial_text](https://github.com/tensorflow/models/tree/master/research/adversarial_text)  
> Preferred Networks & Google & OpenAI, ICLA 2017

#### 工作要点
- 第一个将AT和VAT应用至文本领域
- 结合了FGSM和approximation of LDS，其中前者$L_\infty$约束变为$L_2$约束，即$\epsilon\frac{g}{||g||_2}$
<div class="one-image-container">
    <img src="\AI\Paper_Reading\Trick\Denoising\AdversarialTraining\image\fgm_ablation.png" style="width: 90%;">
    <p style="text-align: center;">FGM效果表现</p>
</div>
- dropout + FGM 效果更好


### TextBugger
> 论文：TextBugger: Generating Adversarial Text Against Real-world Applications  
> Zhejiang University & Alibaba-Zhejiang University Joint Research, NDSS 2019

#### 工作要点
- general attack framework for generating adversarial texts
- 白盒环境：
    - 基于雅可比矩阵，并按照梯度降序argidx排序token序列  
    - 遍历有序token序列，当扰动(char-level或word-level)后$S(x, x^{'})\gt \epsilon\ \text{and} \ F(x^{'})\ne y$，返回$x^{'}$
        - insert、delete、swap、subsitute-char and subsitute-word
        - algorithm 2
- 黑盒环境：选择重要性最高的句子，利用打分函数定位并操纵重要tokens
        - algorithm 3

### TextAttack
TextAttack: A Framework for Adversarial Attacks, Data Augmentation, and Adversarial Training in NLP

### PGD
> 论文：Towards Deep Learning Models Resistant to Adversarial Attacks  
> PGD：**P**rojected **G**radient **D**escent  
> Github：[mnist_challenge](https://github.com/MadryLab/mnist_challenge)、[cifar10_challenge](https://github.com/MadryLab/cifar10_challenge)  
> MIT, ICLA 2018

#### 工作要点
- saddle point (min-max) formulation，通过“小步走，走多步”的鞍点min-max优化方式（FGSM和FGM的优化路径是线性），非线性模型只做一次下降是不够的
- $r_{adv|t+1}=\alpha \frac{g_t}{||g_t||_2}$
- epsilon: maximum perturbation, alpha: step size, steps: number of steps  
- https://blog.csdn.net/weixin_44750512/article/details/132088186  
- https://i-blog.csdnimg.cn/blog_migrate/a43c7bd208f48f266acbb6f64f36e2a5.png  
- start: 是否从随机点开始扰动 `adv_img = img + torch.empty_like(x).uniform_(-eps, eps)`
- 每次更新需要用l∞约束扰动规模，即 `delta=torch.clamp(delta, min=-eps, max=eps), adv_img = img+delta`
- l∞约束比l2约束效果更好，因为后者效果太强了，脱离了对抗的范围，即对抗生成的标签较原真实标签发生了改变  
- https://github.com/Harry24k/adversarial-attacks-pytorch/blob/cf21e81f3f2e7a859e029e3d7953290ef063d6dd/torchattacks/attacks/pgd.py#L74  
- [PGD L2/infinity](https://blog.csdn.net/Sankkl1/article/details/134215790)  
- 不直接输入x_embedding，把fgsm在同一batch内计算n次后（连乘更新），进行min loss，从而解决non-convex（非凸）和non-concave（非凹）点优化问题
- pgd attack，运用局部一阶信息进行强力对抗攻击
- MNIST和CIFAR10白盒攻击效果测试分别为89%/46%，黑盒效果为95%/64%
- Reliable Evaluation of Adversarial Robustness with an Ensemble of Diverse 
Parameter-free Attacks
- l∞ ball around 范式约束
#### 主要内容

### FreeAT
Adversarial Training for Free!
- https://zhuanlan.zhihu.com/p/103593948

### YOPO
You Only Propagate Once


### FreeLB
Free Large-Batch

### SMART
> 论文：SMART: Robust and Efficient Fine-Tuning for Pre-trained Natural Language Models through Principled Regularized Optimization  
> SMAR$^3$T$^2$：**SM**oothness-inducing **A**dversarial **R**egularization and b**R**egman p**R**oximal poin**T** op**T**imization  
> Microsoft Dynamics 365 AI, ACL 2020


#### 工作要点
1. smoothness-inducing regularization
        - https://github.com/archinetai/vat-pytorch/blob/main/vat_pytorch/smart.py  
        - $\lambda_s$LDS + $\mu$PGD  
        - LDS: 对称的KL散度，即KL(P||Q) + KL(Q||P)  
        - PGD使用approx of LDS更新x_adv（每次需要l_infity norm），n次后计算sym_kl_loss  
        - loss = ce_loss + λ*sym_kl_loss
2. bregman proximal point optimization method (including vinilla, generalized, accelerated proximal and other variants)introduce a trust-region-type regularization at each iteration
        - reltaed to FreeLB  
        - vanilla Bregman proximal point(VBPP)，$\theta_{t+1}=\argmin_{\theta} \mathcal{F}(\theta) + \mu\mathcal{D}_{\text{Breg}}(\theta, \theta_t)$    
        - $\mathcal{D}_{\text{Beeg}}(\theta, \theta_t)=\frac{1}{n}\sum_i^n\mathcal{l}_s(f(x_i;\theta), f(x_i; \theta_t))$  
        - $\mathcal{l}_s$，分类：pgd + LDS，回归：pgd + MSE  
        - momentum Bregman proximal point (MBPP)
#### 主要内容