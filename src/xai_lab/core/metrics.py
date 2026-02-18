from __future__ import annotations

import torch


@torch.no_grad()
def confusion_matrix(pred: torch.Tensor, target: torch.Tensor, num_classes: int) -> torch.Tensor:
    """
    Build a confusion matrix (num_classes x num_classes).

    Rows: true class
    Cols: predicted class
    """
    cm = torch.zeros((num_classes, num_classes), dtype=torch.long)
    for t, p in zip(target.view(-1), pred.view(-1)):
        cm[int(t), int(p)] += 1
    return cm


@torch.no_grad()
def precision_recall_f1_from_cm(cm: torch.Tensor):
    """
    Compute per-class precision/recall/F1 and macro-F1 from confusion matrix.

    Why macro-F1?
      With imbalanced data, accuracy can look OK while minority classes are ignored.
      Macro-F1 weights each class equally.
    """
    tp = cm.diag().to(torch.float32)
    fp = cm.sum(dim=0).to(torch.float32) - tp
    fn = cm.sum(dim=1).to(torch.float32) - tp

    precision = tp / (tp + fp + 1e-8)
    recall = tp / (tp + fn + 1e-8)
    f1 = 2 * precision * recall / (precision + recall + 1e-8)

    macro_f1 = f1.mean().item()
    return precision, recall, f1, macro_f1
