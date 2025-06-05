由于模型规模过大，常规情况下的算力无法支持全量微调，因此需要其它部分微调方法（即高效参数微调法PEFT，Parameter-Effecient Fine-Tuning）实现模型的transfer learning。

### PEFT
[LLM-SFT](https://github.com/datawhalechina/self-llm)

```python
from peft import LoraConfig, TaskType, get_peft_model, PeftModel
from transformers import trainer

config = LoraConfig(
    task_type=TaskType.CAUSAL_LM, 
    target_modules=["c_attn", "c_proj", "w1", "w2"],
    inference_mode=False,   # 是否为训练模式
    r=8,                    # Lora 秩
    lora_alpha=32,          # Lora alaph，具体作用参见 Lora 原理
    lora_dropout=0.1        # Dropout 比例
)
model = get_peft_model(model, config)
# 加载lora 权重，需要置lora_config.inference_mode=True
# model = PeftModel.from_pretrained(pre_trained_model, model_id=lora_path, config=lora_config)

print(config, model.print_trainable_parameters(), sep='\n')
# 打印每个可训练参数的名称和形状
for name, param in model.named_parameters():
    if param.requires_grad:
        print(f"Parameter name: {name}, Shape: {param.shape}")

args = TrainingArguments(
    output_dir="./output/Qwen",         # lora_checkpoint 存储路径
    save_steps=100,                     # lora_checkpoint 存储间隔步数
    per_device_train_batch_size=8,      # batch_size
    num_train_epochs=3,                 # epoch
    gradient_accumulation_steps=2,      # 梯度累计步数
    logging_steps=10,                   # 日志打印间隔步数
    gradient_checkpointing=True,
    learning_rate=1e-4,
    save_on_each_node=True
)

trainer = Trainer(
    model=model,
    args=args,
    train_dataset=tokenization,     # 具体的tokenization方案
                                    # tokenization = dataset.map(preprocess_fn,
                                    #    batched=True,
                                    #    batch_size=10)
    data_collator=DataCollatorForSeq2Seq(tokenizer=tokenizer, padding=True),
)

trainer.train()

# inference
ipt = tokenizer(prompt.format(input_text) + prompt_details, return_tensors="pt").to("cuda")
print(tokenizer.decode(model.generate(
    **ipt,
    do_sample=True,
    top_k=1,
    eos_token_id=tokenizer_eos_token_id),
    skip_special_tokens=True)
)
```

#### [LoRA](lora.md)
- [Mixture of LoRA Experts (MoLE)](https://openreview.net/pdf?id=uWvKBCYh4S)
- [Higher Layers Need More LoRA Experts](https://arxiv.org/pdf/2402.08562v1)
- [LoRA insight experiments](https://lightning.ai/pages/community/lora-insights/#toc12)
- [Practical Tips when using LoRA](https://magazine.sebastianraschka.com/p/practical-tips-for-finetuning-llms)
- [LoRA servey](https://towardsdatascience.com/an-overview-of-the-lora-family-515d81134725)

#### [AdaLoRA](adalora.md)

#### [PiSSA](pissa.md)


#### [DoRA](dora.md)

#### [LoRA-GA](lora-ga.md)

#### [QLoRA](qlora.md)

- https://towardsdatascience.com/an-overview-of-the-lora-family-515d81134725