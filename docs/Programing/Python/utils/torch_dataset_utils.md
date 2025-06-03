```python
from torch.utils.data import Dataset
import os
from tqdm import tqdm
from wheel_utils.general_dataset_utils import *


class MyDataset(Dataset):
    def __init__(
            self,
            data_file,
            trie=None,
            t2s=None,
            case_sensitive=False
    ):
        self.trie = trie
        self.t2s = t2s
        self.case_sensitive = case_sensitive
        self.data, self.labels = self._read_data_file(data_file)

    def _read_data_file(self, data_file):
        base_name = os.path.basename(data_file)
        data, labels = [], []
        with open(data_file, 'r', encoding='utf-8') as f:
            for line in tqdm(f, desc=f"parsing {base_name}"):
                try:
                    line = json.loads(line)
                except:
                    print(json.dumps(line, ensure_ascii=False))
                    raise ValueError

                line = pre_process_content(line, self.trie, self.t2s, self.case_sensitive)
                label = uni_label(line.get("label", "0"))
                data.append(line["content"])
                labels.append(label)

        print(f"{base_name} has {len(data)} samples")
        return data, labels

    def __len__(self):              # 重写__len__魔法方法
        return len(self.data)

    def __getitem__(self, idx):     # 重写__item__数据访存方法
        return self.data[idx], self.labels[idx]
```


### DataLoader
```python
from torch.utils.data import DataLoader

```

### datasets
```python
from datasets import load_dataset

```