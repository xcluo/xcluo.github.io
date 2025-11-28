## CoT
> 论文：**C**hain-**o**f-**T**hought Prompting Elicits Reasoning in Large Language Models  
> Google Brain, 2022 Jan, NeurIPS 2022

### 主要内容


## CoT-SC
> 论文：**S**elf-**C**onsistency Improves Chain of Thought Reasoning in Language Models  
> Google Research & Brain Team, 2022 Mar, ICLR 2023

### 主要内容
SC的核心思想为取代贪婪解码的‘一条路走到黑’，通过‘集思广益’来得到更可靠的答案。即

- 一个问题和CoT示例，K次输入得到K个推理结果（传统CoT只会贪婪解码，选择最可能的答案）
- 统计CoT示例生成的的K个推理答案。选择出现频数最高的答案作为最终输出。

> 基于多次投票，规避单次错误

## ToT
> 论文：**T**ree **o**f **T**houghts: Deliberate Problem Solving with Large Language Models  
> Princeton University & Google DeepMind, 2023 May, NeurIPS 2023

### 主要内容

## GoT
> 论文：**G**raph **o**f **T**houghts: Solving Elaborate Problems with Large Language Models  
> ETH Zurich & Warsaw University of Technology & Cledar, 2023 Aug, AAAI 2024

### 主要内容