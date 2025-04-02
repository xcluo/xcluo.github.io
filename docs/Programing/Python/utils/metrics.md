```python
from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score
import numpy as np


def compute_metrics(logit_label_pairs):
    logits, labels = logit_label_pairs
    predictions = np.argmax(logits, axis=-1)
    return {
        "accuracy": accuracy_score(labels, predictions),
        "f1": f1_score(labels, predictions, average="macro"),
        "recall": recall_score(labels, predictions),
        "precision": precision_score(labels, predictions),
    }
```