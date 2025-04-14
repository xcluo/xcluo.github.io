#
[Dataset]()
```python
from torch.utils.data import Dataset, DataLoader
```
### torch生态库
#### torchtext
`pip install torchtext`
#### torchvision
`pip install torchvision`

```python
from torchvision.utils import save_image, transforms


transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=(0.5, ),  # 3 for RGB channels
                         std=(0.5, ))])
   
def denorm(x):
    out = (x + 1) / 2
    return out.clamp(0, 1)

fake_images = generate_image.view(generate_image.size(0), 1, 28, 28)
save_image(denorm(fake_images.data), 'samples/test.png')
```

#### torchaudio
`pip install torchaudio`
#### torchmetrics
`pip install torchmetrics`

=== "Precision/Recall"

    ```python
    class Precision/Recall(StatScores):
        def __init__(
            self,
            num_classes: Optional[int] = None,
            threshold: float = 0.5,
            average: str = "micro",
            mdmc_average: Optional[str] = None,
            ignore_index: Optional[int] = None,
            top_k: Optional[int] = None,
            multiclass: Optional[bool] = None,
            compute_on_step: Optional[bool] = None,
            **kwargs: Dict[str, Any],
        )
    ```

=== "F1Score"

    ```python
    class F1Score(FBetaScore):
        def __init__(
            self,
            num_classes: Optional[int] = None,          # num_of_classes
            threshold: float = 0.5,                     # prob ≥ threshold → True
            average: str = "micro",                     # micro: 全局平均
                                                        # macro: 加权平均
                                                        # ...
            mdmc_average: Optional[str] = None,
            ignore_index: Optional[int] = None,         # 计算时忽略指定的标签值
            top_k: Optional[int] = None,                # 返回目标类置信度最高的top-k个
            multiclass: Optional[bool] = None,
            compute_on_step: Optional[bool] = None,
            **kwargs: Dict[str, Any],
        )
    ```

=== "Accuracy"

    ```python
    class Accuracy(StatScores):
        def __init__(
            self,
            threshold: float = 0.5,                     # prob ≥ threshold → True
            num_classes: Optional[int] = None,          # num_of_classes
            average: str = "micro",                     # micro: 全局平均
                                                        # macro: 加权平均
                                                        # ...
            mdmc_average: Optional[str] = "global", 
            ignore_index: Optional[int] = None,         # 计算时忽略指定的标签值
            top_k: Optional[int] = None,                # 返回目标类置信度最高的top-k个
            multiclass: Optional[bool] = None,
            subset_accuracy: bool = False,
            compute_on_step: Optional[bool] = None,
            **kwargs: Dict[str, Any],
        )
    ```

```python
from torchmetrics import Accuracy, F1Score, Precision, Recall


# 初始化指标（支持多分类、多标签）
accuracy = Accuracy(num_classes=2)
precision = Precision(num_classes=2)
recall = Recall(task="multiclass", num_classes=10, average='macro')

# 计算批次指标
logits = torch.randn(32, 10)  # batch_size=32, 10类
labels = torch.randint(0, 10, (32,))
preds = torch.argmax(logits, dim=1).to("cpu")

# accuracy.update(preds, labels)
# precision.update(preds, labels)
# recall.update(preds, labels)

# 获取累积结果
print(f"Accuracy: {accuracy.compute():.4f}")
print(f"Precision: {precision.compute():.4f}")
print(f"Recall: {recall.compute():.4f}")

# 重置指标
accuracy.reset()


torchmetrics.functional.accuracy(preds, target)

for s in batch_s:
    accuracy(preds, target)     # batch accuracy
accuracy.compute()              # total accuracy

```