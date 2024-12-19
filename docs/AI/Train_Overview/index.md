#### Pre-process
**经典算法细节处理**：

1. ^^数字形式归一化^^，如 `① -> １`
2. ^^空白字符归一化^^，如 `[SPACE]` 或 `" "`
    - 前者在空白字符表语义时使用，后者在空白字符无语义时使用
3. ^^emoji处理^^，如正向 `🙂 -> [微笑]` 或逆向 `[微笑] -> 🙂`
    - 前者直接解析emoji，后者基于`emoji_embedding`
4. ^^字符上下标去除^^：如 [BasicTokenizer._run_strip_accents](/Programing/Python/ai_libs/bpe_tokenizer/#_1)

#### Data Augmentation

#### Tokenization
- char-tokenization: 可较好地适应变种表述方式
<div class="admonition info" style="margin-left: 20px;">
    <!-- <p class="admonition-title"></p> -->
    <ol>
        <li><a href="\AI\Paper_Reading\Trick\Multimodality_Fusion#sound-modality">sound_modality</a>：<span style="color:red;">为防止发散，无法获取拼音的token应保留原值</span></li>
        <li> 由于 <code>pad_to_max_length</code> 是给所有短的部分用0填充（因此词表0-th词一般为[PAD]），为防止不同模态的[PAD]字符语义混淆，各模态的词表和embedding_table应互相独立</li>
        <li><span style="color:green;">处理任意长度样本</span>：不在分词阶段(前向或后向)截断过长字符串，而是整体tokenization并输入，只通过过滤逻辑控制输入样本长度</li>
        <li>shape_modality</li>
    </ol>
</div>  




#### Embedding

#### Ensemble & MoE

#### Distillation