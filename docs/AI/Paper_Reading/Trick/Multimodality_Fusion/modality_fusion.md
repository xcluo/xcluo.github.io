### Modality Fusion
#### train

1. 多流输入`input_modality_i`
2. 多流输出`output_modality_i`
3. 融合多模态`fusion=concat(modality_1, ..., modality_n)`
4. $\mathcal{L} = \sum \mathcal{L}(output_{modality_i}) + \mathcal{L}(fusion)$

#### infer

 1. 多流输入`input_modality_i`
 2. 多流输出`output_modality_i`
 3. 融合多模态`fusion=concat(modality_1, ..., modality_n)`
 4. `max(output_modality_i_prob, ..., fusion_prob)`


### Attentions
#### embedding/state concat
1. embedding concat

    为了更好地使各模态充分融合交织，选择在embedding层进行concat操作，此时：

       - 仅保留融合更为彻底的fusion作为优化目标，即$\mathcal{L}(fusion)$

2. hidden_state concat

    一般为了充分学习好各模态的embedding，会选择在最后的投影层进行concat操作，此时：

      - 既可以使用fusion作为优化目标，即$\mathcal{L}(fusion)$
      - 又能保留各模态作为优化目标，即$\sum \mathcal{L}(output_{modality_i})$

!!! info
    最好是进行`concat`操作而不是<span style="color:red;">简单的线性相加</span>，后者<span style="color:red;">会随着非线性操作的增加而模糊各模态的特征区别导致发散</span>。