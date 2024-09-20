#### 蒸馏技巧

1. 软标签蒸馏：使用教师模型的软标签（各类别概率）进行蒸馏
2. 蒸馏时init with pretrained word_embeddings
    - 快速学习
    - 未加入训练的token知识也能从init_word_embeddings中迁移得到，大幅减少参与训练的正常样本数
3. 蒸馏迭代时，样本选择  
      1. 各类别中前后标签不一致的  
      2. 以$abs(prob_{pre} - prob_{cur})$降序排序