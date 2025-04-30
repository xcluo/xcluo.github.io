transformers版本越新越好，且要求python>=3.11

- [ ] transformers'
- [ ] from transformers import Trainer, TrainingArguments
- [ ] transformers.optimization

```python
from transformers import Trainer, TrainingArguments, AutoTokenizer, AutoModelForSequenceClassification
from datasets import load_dataset
from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score
import numpy as np
from opencc import OpenCC
from wheel_utils.char_alpha_numeric import *
from sklearn.cluster import KMeans


trie = StringUtils.SpanReplacement("../data/replace_map.txt")
t2s = OpenCC("t2s")


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return {
        "accuracy": accuracy_score(labels, predictions),
        "f1": f1_score(labels, predictions, average="macro"),
        "recall": recall_score(labels, predictions),
        "precision": precision_score(labels, predictions),
    }


def tokenize_function(examples):
    return tokenizer(
        examples["content"],
        padding="longest",
        truncation=True,
        max_length=max_seq_length,
        # return_tensors="pt"
    )


def uni_label(label):
    if label in {'0', 0, 'normal', 'other'}:
        return 0
    elif label in {'1', 1, 'porno', '2'}:
        return 1
    else:
        raise ValueError(f"{label} is not a valid label")


def uni_labels(example):
    example["label"] = uni_label(example.get("label", "0"))
    return example


def pre_process_content(example):
    cnt = example.get("content", example.get("Content", example.get("c")))
    cnt_no_emoji = trie.replace_span(cnt)
    example['content'] = cnt_no_emoji.lower()
    return example


def pre_process(example):
    example = uni_labels(example)
    example = pre_process_content(example)
    return example


def split_dataset(data_files, train_percent=.99):
    dataset_dict = load_dataset("json", data_files)
    if len(dataset_dict) == 1:
        dataset = dataset_dict["train"].shuffle()
        train_dataset = dataset.select(range(int(train_percent * len(dataset))))
        valid_dataset = dataset.select(range(int(train_percent * len(dataset)), len(dataset)))
    else:
        train_dataset = dataset_dict["train"]
        valid_dataset = dataset_dict["valid"]
    train_dataset = train_dataset.map(pre_process, desc="preprocessing train dataset")
    valid_dataset = valid_dataset.map(pre_process, desc="preprocessing valid dataset")
    tokenized_train_dataset = train_dataset.map(tokenize_function, batched=True, desc="tokenizing train dataset")
    tokenized_valid_dataset = valid_dataset.map(tokenize_function, batched=True, desc="tokenizing valid dataset")

    return tokenized_train_dataset, tokenized_valid_dataset


if __name__ =="__main__":
    data_path = {"train": "../data/data.json"}
    model_path = "../../pre_trains/roberta/"
    save_steps = 1000
    gradient_accumulation_steps = 1
    train_batch_size = 64
    eval_batch_size = 64
    max_seq_length = 256
    epochs = 1
    lr=2e-5

    # init model
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSequenceClassification.from_pretrained(model_path)

    # prepare dataset
    tokenized_train_dataset, tokenized_valid_dataset = split_dataset(data_path)

    # trainer arguments
    training_args = TrainingArguments(
        # dataset part
        remove_unused_columns=True,
        # seed=42,
        # group_by_length=False,
        # dataloader_num_workers=4,

        # train part
        learning_rate=lr,
        lr_scheduler_type="linear",
        warmup_ratio=0.1,
        # warmup_steps=100,
        optim="adamw_hf",
        num_train_epochs=epochs,
        per_device_train_batch_size=train_batch_size,
                                        # 等价于`per_device_train_batch_size`，推荐使用前者
                                        # 因此真实batch_size=num_gpu * per_device_train_batch_size
        gradient_accumulation_steps=gradient_accumulation_steps,
        # gradient_checkpointing=True,

        # eval part
        per_device_eval_batch_size=eval_batch_size,
        evaluation_strategy="steps",
        eval_steps=save_steps,
        metric_for_best_model="f1",
        greater_is_better=True,         # 上述metric指标是否越大越好

        # dump part
        save_strategy="steps",
        save_steps=save_steps,
        save_total_limit=2,             # 目标文件夹最多dump_model数（包括历史训练结果）
        output_dir="./ckpts/",
        load_best_model_at_end=False,   # 防止无法save_at_end
        # save_at_end=True              # Transfomrer≥4.36.0才应用

        # log part
        logging_dir="./logs/",
        logging_steps=100,
    )

    trainer = Trainer(
        model,
        training_args,
        train_dataset=tokenized_train_dataset,
        eval_dataset=tokenized_valid_dataset,
        compute_metrics=compute_metrics,
        tokenizer=tokenizer
    )

    trainer.train()
    trainer.save_model("./ckpts/final_model/")  # 训练结束后调用
```