- state: 当前全局状态  
- observation：用户当前能观察到的状态  
- agent：执行动作的主体  
- action：agent的动作，一般含有action space，为集合，如超级玛丽中的{左、右、跳、打、保持静止}  
- reward：agent执行完action后环境给agent的奖励或惩罚  
- environment：  
- policy：策略函数$\pi(action_t \vert s_t)$  
- trajectory：轨迹，用$\tau$表示，即一连串的状态和动作序列$\{s_0, a_0, s_1, a_1, \cdots\}$，其中确定状态是$s_{t+1}=f(s_t, a_t)$，随机状态$s_{t+1}=P(\cdot\vert s_t, a_t)$  
- return: 从当前状态到游戏结束reward的总和
- $R(\tau^{n}_t) = \sum_{t^{'}=t}^{T_n}\gamma^{t^{'} - t}\times r_{t^{'}}^n$  
- action-value function，在当前状态下，做出动作后的期望奖励$Q_{\theta}(s, a)$  
- state-value function，在当前状态下，期望的回报$V_{\theta}(s)$  
- advantage function，在当前状态下，做出动作后比期望汇报带来了多少优势，即$A_{\theta}(s, a) = Q_{\theta}(s, a) - V_{\theta}(s)$，替换$R_{t}^{n}-B(s^n_t)$ baseline基础偏移防止坏的情况都坏，好的情况都好现象  
- 由于目标是最大化 $R(\tau)P_{\theta}(\tau)$，因此需要使用梯度上升方法进行更新

核心区别在于数据可不可以重复利用

- on-policy  
    - $E_{\tau \sim p_{\theta}(\tau) }[R(\tau)\nabla \log p_{\theta}(\tau)]$  
    - 边学习数据边更新状态，此时需要使用新状态下的数据，数据利用率低，

- off-policy  
    - $E_{\tau \sim p_{\theta}^{'}(\tau) }[\frac{p_\theta(\tau)}{p_{\theta}^{'}(\tau)}R(\tau)\nabla \log p_{\theta}(\tau)]$  
    - 观察他人学习，数据利用率高

- IS: importance sampling  
- $\frac{p_\theta(\tau)}{p_{\theta}^{'}(\tau)}$ 为重要性权重 importance weight

--------
- q-learning
- policy gradient methods（PG）
- trust region policy optimization (TRPO)
- ppo  
    - Generalized Advantage Estimation, GAE，多步采样方式结合，步数越多权重越小

    <div class="one-image-container">
        <img src="image/ppo.png " style="width: 80%;">
        <!-- <p>LoRA在Attention各部分权重上的消融实验效果</p> -->
        <!-- <figcaption>这是图片的标题或描述。</figcaption> -->
    </div>
- dpo
- grpo
    - [Approximating kl divergence](http://joschu.net/blog/kl-approx.html), seq-all-token prob
    - tokne-level 输出

- actor model，用于RLHF学习更新

- reward model
    - reward function: xmlcount_reward_func, soft_format_reward_func, strict_format_reward_func, int_reward_func, correctness_reward_func
    - pair-wise loss：比较对象的相对偏好来优化模型，如$\log \sigma(f(x_i) - f(x_j))$
    - point-wise loss：分类或回归问题

- (state) value model
- performance/reference model
- active/actor/policy model

#### llm sft
- [ ] [大模型监督微调SFT](https://www.bilibili.com/video/BV1gmWDeLEMZ?spm_id_from=333.788.videopod.sections&vd_source=782e4c31fc5e63b7cb705fa371eeeb78)
    - chat template
    - completion only，只对输出结果计算loss，不计算通用chat template的loss（如通过completion mask进行标记）
    - NEFTune, noisy embeddings finetuning: 对输入embedding加了一点噪音，sqrt(seq_len * dim) 是为了归一化，防止由于每次input长度不同导致的欧式距离不同，可与attention权重分数类比

#### llm rl

- question，问题
- chosen，更好的回答
- rejected，相对不好的回答
- [grpo trainer](https://huggingface.co/docs/trl/main/en/grpo_trainer)