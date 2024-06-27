#### Pre-process
**经典算法细节处理**：

1. ^^数字形式归一化^^，如 `① -> １`
2. ^^空白字符归一化^^，如 `[SPACE]` 或 `""`
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
        <li><a href="\AI\Paper_Reading\Trick\Multimodality_Fusion#sound-modality">sound_modality</a>：<span style="color:red;">为防止发散，最好只对汉字、数字和字母进行pinyin化，其他的字符用 <code>[Sound_PAD]</code> 和 <code>[Sound_UNK]</code> 统一表示</span></li>
        <li> 由于 <code>pad_to_max_length</code> 是给所有短的部分用0填充（因此词表第一个词一般为[PAD]），为防止不同模态的[PAD]字符语义混淆，各模态的词表和embedding_table应互相独立</li>
        <li>shape_modality</li>
    </ol>
</div>  




#### Embedding

#### Ensemble & MoE

#### Distillation