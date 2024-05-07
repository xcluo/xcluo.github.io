#### Data Augmentation

#### Tokenization
- char-tokenization: 可较好地适应变种表述方式
<div class="admonition info" style="margin-left: 20px;">
    <!-- <p class="admonition-title"></p> -->
    <ol>
        <li>sound_modality</li>
        <li>shape_modality</li>
    </ol>
    <span style="color:red;">注意每种模态embedding初始化时mean和std要互相持平，否则可能导致（某一模态占比过重而）发散</span>
</div>  

- bpe-tokenization: 基于统计方法将单词划分为字词，更好地表述词的相关性
!!! info
    经典细节处理：

    1. 替换，如 `① -> １`
    2. 替换空白字符 `replace_white_space` with `[SPACE]` or `""`
    3. emoji处理，如 `🙂 -> [微笑]`


#### Embedding

#### Ensemble & MoE

#### Distillation