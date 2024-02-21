### Python
```python title="json内容去重"
import argparse
import json
from tqdm import tqdm


parser = argparse.ArgumentParser()
parser.add_argument('-f', '--file_name', type=str, required=True, help='var description')
args = parser.parse_args()  # 通过`args.var_name` 调用相应参数


if __name__ == "__main__":
    dump_lines = set()
    with open(args.file_name, 'r', encoding='utf-8') as f, \
            open('./unique_result.json', 'w', encoding='utf-8') as fout:
        for line in tqdm(f):
            line = json.loads(line)
            cnt = line.get('content', line.get('c'))
            if cnt in dump_lines:
                continue
            dump_lines.add(cnt)
            fout.write(json.dumps(line, ensure_ascii=False) + '\n')
            fout.flush()
            if len(dump_lines) >= 30000000:
                dump_lines.clear()
```