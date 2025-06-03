```python
import random
import json
import tqdm
from wheel_utils.char_alpha_numeric import *


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


def pre_process_content(example, trie, t2s, case_sensitive):
    cnt = example.get("content", example.get("Content", example.get("c")))

    if trie:
        cnt = trie.replace_span(cnt)
    if t2s:
        cnt = t2s.convert(cnt)
    if not case_sensitive:
        cnt = cnt.lower()

    cnt = PunctuationUtils.strip_white_space(cnt, replace_token=" ")
    example["content"] = cnt
    return example


def pre_process(example, trie, t2s, case_sensitive):
    example = uni_labels(example)
    example = pre_process_content(example, trie, t2s, case_sensitive)
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


def generate_distill_data(object_file, parts_num=12, pp=8, threshold=.5, label_type="soft", ignore_teacher_data=True):
    tmp_data = set()
    if ignore_teacher_data:
        with open('../data.json', 'r', encoding='utf-8') as f_line:
            for line in f_line:
                line = json.loads(line)
                tmp_data.add(line.get('content', line.get("c")))
    
    neg_num, pos_num = 0, 0
    total_num = 0
    for i in range(1, parts_num + 1):
        with open(f'../{object_file}/result_{i:>02d}.txt', 'r', encoding='utf-8') as f_prob:
            for prob in tqdm(f_prob, f"scan part_{i} positives"):
                prob = json.loads(prob)
                if prob['prob'] >= threshold:
                    pos_num += 1
                total_num += 1
    target_neg_num = pp * pos_num
    neg_keep_rate = target_neg_num / (total_num - pos_num)

    with open('./data_distill.json', 'w', encoding='utf-8') as f:
        for i in range(1, parts_num + 1):
            with open(f'../{object_file}/result_{i:>02d}.txt', 'r', encoding='utf-8') as f_prob, \
                    open(f'../{object_file}/{object_file}{i:>02d}', 'r', encoding='utf-8') as f_line:
                for k, (line, prob) in tqdm(enumerate(zip(f_line, f_prob), 1), f"dump part_{i} samples"):
                    prob = json.loads(prob)
                    if prob['prob'] >= threshold or random.uniform(0, 1) < neg_keep_rate:
                        line = json.loads(line)
                        content = line.get('content', line.get("c"))
                        
                        if content in tmp_data:
                            continue

                        if label_type == "soft":
                            probs = [1 - prob["prob"], prob["prob"]]
                        elif label_type == "hard":
                            probs = [ 1-int(prob["prob"]  >= threshold) , int(prob["prob"]  >= threshold)]
                        else:
                            raise ValueError("label type error")

                        sample = {
                            "content": content,
                            "probs": probs
                        }

                        neg_num += prob['prob'] < threshold
                        f.write(json.dumps(sample, ensure_ascii=False) + "\n")
                        f.flush()

    print(f"distill_data has:\n\t{pos_num} positives\n\t{neg_num} negatives")
```