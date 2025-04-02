```python
import random
import json


def uni_label(label):
    if label in {'0', 0, 'normal', 'other'}:
        return 0
    elif label in {'1', 1}:
        return 1
    else:
        raise ValueError(f"{label} is not a valid label")


def uni_labels(example):
    example["label"] = uni_label(example.get("label", "0"))
    return example


def pre_process_content(trie, t2s, case_sensitive, example):
    cnt = example.get("content", example.get("Content", example.get("c")))

    if trie:
        cnt = trie.replace_span(cnt)
    if t2s:
        cnt = t2s.convert(cnt)
    if not case_sensitive:
        cnt = cnt.lower()

    example["content"] = cnt
    return example


def pre_process(trie, t2s, case_sensitive, example):
    example = uni_labels(example)
    example = pre_process_content(trie, t2s, case_sensitive, example)
    return example


def split_corpus(data_file, train_data_file, valid_data_file, valid_percent=0.1):
    with open(data_file, 'r', encoding='utf-8') as f_in, \
            open(train_data_file, 'w', encoding='utf-8') as f_train, \
            open(valid_data_file, 'w', encoding='utf-8') as f_valid:
        train_part, valid_part = [], []
        for line in f_in:
            line = json.loads(line)
            if random.uniform(0, 1) < valid_percent:
                valid_part.append(line)
            else:
                train_part.append(line)
        random.shuffle(valid_part)
        for line in valid_part:
            f_valid.write(json.dumps(line, ensure_ascii=False) + '\n')
            f_valid.flush()

        random.shuffle(train_part)
        for line in train_part:
            f_train.write(json.dumps(line, ensure_ascii=False) + '\n')
            f_train.flush()

        print(f"after splitting,there are"
              f"\n{len(train_part)} samples in train set"
              f"\n{len(valid_part)} samples in valid set")

```