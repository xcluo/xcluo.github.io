由于模型规模过大，常规情况下的算力无法支持全量微调，因此需要其它部分微调方法（即高效参数微调法PEFT，Parameter-Effecient Fine-Tuning）实现模型的transfer learning。

### PEFT
[LLM-SFT](https://github.com/datawhalechina/self-llm)

```python
from peft import LoraConfig, TaskType, get_peft_model
from transformers import trainer

config = LoraConfig(
    task_type=TaskType.CAUSAL_LM, 
    target_modules=["c_attn", "c_proj", "w1", "w2"],
    inference_mode=False, # 训练模式
    r=8, # Lora 秩
    lora_alpha=32, # Lora alaph，具体作用参见 Lora 原理
    lora_dropout=0.1# Dropout 比例
)
model = get_peft_model(model, config)
# model = PeftModel.from_pretrained(model, model_id=lora_path, config=config)
print(config, model.print_trainable_parameters(), sep='\n')
# 打印每个可训练参数的名称和形状
for name, param in model.named_parameters():
    if param.requires_grad:
        print(f"Parameter name: {name}, Shape: {param.shape}")

args = TrainingArguments(
    output_dir="./output/Qwen",
    per_device_train_batch_size=8,
    gradient_accumulation_steps=2,
    logging_steps=10,
    num_train_epochs=3,
    gradient_checkpointing=True,
    save_steps=100,
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