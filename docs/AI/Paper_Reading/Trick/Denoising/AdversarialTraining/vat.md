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
> Github: [vat_pytorch](https://github.com/lyakaap/VAT-pytorch/blob/master/vat.py#L38)  
> Preferred Networks & Kyoto University & ATR Cognitive Mechanisms Laboratories & Ritsumeikan University, TPAMI 2017

#### 工作要点
- [x] 虚拟对抗训练的半监督算法(VAT, Vitural Adversarial Training)，运用了平滑思想旨在使模型对处于一定的区间范围内的数据样本都有较为相似的分类结果。

    $$
    \begin{aligned}
        \mathcal{L} &= \mathop{\mathcal{L}_1}\limits_{(x_1,y_1) \in D_{label}}(x_1, y_1, \theta) + \alpha \mathop{\text{KL}}\limits_{x_2 \in D_{all}}[f(x_2)||f(x_2+r_{\text{v-adv}})] \\
        r_{\text{v-adv-}2} &= \epsilon\frac{g}{||g||_2}, \text{where}\ g=\nabla_{r}\text{KL}[p(y|x_2,\theta)||p(y|x_2+r,\theta)]\Big\vert_{r=\epsilon d} \\
        r_{\text{v-adv-}\infty} &= \epsilon \text{sign}(g), \text{where}\ g=\nabla_{r}\text{KL}[p(y|x_2,\theta)||p(y|x_2+r,\theta)]\Big\vert_{r=\epsilon d}
    \end{aligned}
    $$

    !!! info ""
        - 局部平滑目标，即扰动范围$||r||_{2/\infty}\le \epsilon$内，$f(x)\approx f(x+r)$  
        - 两个超参 $\epsilon$ 和 $\alpha$，以及指定norm方式  
        - $(x_1,y_1)\in D_{label}, x_2\in D_{all}$，前者有监督训练，后者无监督局部分布平滑，因此为半监督训练
        - $\nabla_r$中的$r$表示的是`random_init_r`
        - 单次训练需要3次前向计算：`forward(x), forward(x2_update_r), forward(x2_get_final_kl)`
        - 单次后向需要2次后向计算：`backward(x2_update_r), backward(update_θ)`

- [x] VAT较（使用伪标签的）对抗训练模型的泛化能力优秀（AT新增训练点，VAT直接泛化面）

#### 主要内容
1. approximation of LDS

    $$
    \begin{aligned}
        \text{LDS}(r, x, \theta) =& \text{KL}(r, x, \theta)=\text{KL}[f(x)||f(x+r)] \\
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
        - $\text{arg max}\text{ KL}$ 表示选用$r$使得KL值最大参与VAT训练（此时认为对抗效果最好）  
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
- [x] 第一个将AT和VAT应用至文本领域
- [x] 结合了FGSM和approximation of LDS，其中前者$L_\infty$约束变为$L_2$约束，即$\epsilon\frac{g}{||g||_2}$
<div class="one-image-container">
    <img src="\AI\Paper_Reading\Trick\Denoising\AdversarialTraining\image\fgm_ablation.jpg" style="width: 90%;">
    <!-- <p style="text-align: center;">FGM效果表现</p> -->
</div>
- [x] dropout + FGM 顺序搭配使用效果更好


### PGD
> 论文：Towards Deep Learning Models Resistant to Adversarial Attacks  
> PGD：**P**rojected **G**radient **D**escent  
> Github：[mnist_challenge](https://github.com/MadryLab/mnist_challenge)、[cifar10_challenge](https://github.com/MadryLab/cifar10_challenge)  
> MIT, ICLA 2018

#### 工作要点
- 不同于**FG(S)M**先验地认为既有模型为简单线性分类器，此类对抗训练复杂模型效果不够适用（局部最优或效果不理想），因此提出了一个**优化的多步变体方案**适用于复杂模型的对抗训练
    <div class="one-image-container">
        <img src="\AI\Paper_Reading\Trick\Denoising\AdversarialTraining\image\pgd_diagram.png" style="width: 30%;">
        <p style="text-align: center;">PGM原理示意</p>
    </div>

    $$
    \begin{aligned}
        \mathcal{L}(x, y, \theta) &= \mathcal{L}_1(x, y, \theta) + \alpha\mathcal{L}_1(x^S, y, \theta) \\
        x^{t+1} &= x^0 + r^t \\
        r^{t+1}_{\infty}& = \epsilon\text{sign}(\nabla_x\mathcal{L}_1(x^t, y, \theta)) \\
        r^{r+1}_{2}& = \epsilon\frac{\nabla_x\mathcal{L}_1(x^t, y, \theta)}{||\nabla_x\mathcal{L}_1(x^t, y, \theta)||_2} 
    \end{aligned}
    $$

    !!! info ""
        - 为增加样本随机性，可初始化`x0=x+clip(random_r, delta)`作为PGD起始点  
        - $S$ 表示迭代的步数，可通过`clip(r, delta)`方案控制每步新增的$r$
    



### SMART
> 论文：SMART: Robust and Efficient Fine-Tuning for Pre-trained Natural Language Models through Principled Regularized Optimization  
> SMAR$^3$T$^2$：**SM**oothness-inducing **A**dversarial **R**egularization and b**R**egman p**R**oximal poin**T** op**T**imization  
> Github：[mt-dnn](https://github.com/namisan/mt-dnn/blob/master/mt_dnn/perturbation.py)，[smart_pytorch](https://github.com/archinetai/smart-pytorch/blob/main/smart_pytorch/smart_pytorch.py#L43)  
> Microsoft Dynamics 365 AI, ACL 2020


#### 工作要点

- [x] 结合approximation of LDS和PGD思想

    $$
    \begin{aligned}
        \mathcal{L} = \mathcal{L}_1(x, y, \theta) &+ \alpha \big(\text{KL}[f(x)||f(x^S)] + \text{KL}[f(x^S)||f(x)]\big) \\ 
        x^{t+1} &= x^0 + r^t \\
        r^{t+1}_{\text{v-adv-}2} &= \epsilon\frac{g}{||g||_2}, \text{where}\ g=\nabla_{r}\text{KL}[f(x)||f(x^0 + r)] \Big\vert_{r=r^t} \\
        r^{t+1}_{\text{v-adv-}\infty} &= \epsilon \text{sign}(g), \text{where}\ g=\nabla_{r}\text{KL}[f(x)||f(x^0 + r)] \Big\vert_{r=r^t}
    \end{aligned}
    $$

    !!! info ""
        - 此处approximation of LDS中使用了对称KL散度
        - $\nabla_r$中的$r$表示的是`random_init_r`

- [x] SMART较正常BERT训练效果提升
<div class="one-image-container">
    <img src="\AI\Paper_Reading\Trick\Denoising\AdversarialTraining\image\smart_ablation.jpg" style="width: 70%;">
    <!-- <p style="text-align: center;">SMART效果表现(-$\mathcal{R}_s$表示去除LDS部分)</p> -->
</div>


------

1 TextBugger
> 论文：TextBugger: Generating Adversarial Text Against Real-world Applications  
> Zhejiang University & Alibaba-Zhejiang University Joint Research, NDSS 2019

1 工作要点
- general attack framework for generating adversarial texts
- 白盒环境：
    - 基于雅可比矩阵，并按照梯度降序argidx排序token序列  
    - 遍历有序token序列，当扰动(char-level或word-level)后$S(x, x^{'})\gt \epsilon\ \text{and} \ F(x^{'})\ne y$，返回$x^{'}$
        - insert、delete、swap、subsitute-char and subsitute-word
        - algorithm 2
- 黑盒环境：选择重要性最高的句子，利用打分函数定位并操纵重要tokens
        - algorithm 3

1 TextAttack
TextAttack: A Framework for Adversarial Attacks, Data Augmentation, and Adversarial Training in NLP

- Reliable Evaluation of Adversarial Robustness with an Ensemble of Diverse Parameter-free Attacks

1 FreeAT
Adversarial Training for Free!
- https://zhuanlan.zhihu.com/p/103593948

1 YOPO
You Only Propagate Once


1 FreeLB
Free Large-Batch