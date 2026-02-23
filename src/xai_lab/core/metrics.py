from __future__ import annotations

import torch


@torch.no_grad()
def confusion_matrix(
    predicted_labels: torch.Tensor,
    true_labels: torch.Tensor,
    num_classes: int,
) -> torch.Tensor:
    """
    Build a confusion matrix (num_classes x num_classes).

    Rows: true class
    Cols: predicted class
    """
    confusion = torch.zeros((num_classes, num_classes), dtype=torch.long)

    # Flatten in case tensors come in as [B, ...]
    true_flat = true_labels.view(-1)
    pred_flat = predicted_labels.view(-1)

    for true_class, pred_class in zip(true_flat, pred_flat):
        confusion[int(true_class), int(pred_class)] += 1

    return confusion


@torch.no_grad()
def precision_recall_f1_from_cm(confusion: torch.Tensor):
    """
    Compute per-class precision/recall/F1 and macro-F1 from a confusion matrix.

    Why macro-F1?
      With imbalanced data, accuracy can look OK while minority classes are ignored.
      Macro-F1 weights each class equally.
    """
    # True positives are on the diagonal
    true_positives = confusion.diag().to(torch.float32)

    # Totals by predicted and actual class
    predicted_totals = confusion.sum(dim=0).to(torch.float32)  # column sums
    actual_totals = confusion.sum(dim=1).to(torch.float32)     # row sums

    # Derive FP and FN from totals
    false_positives = predicted_totals - true_positives
    false_negatives = actual_totals - true_positives

    # Per-class metrics (add epsilon to avoid divide-by-zero)
    precision_per_class = true_positives / (true_positives + false_positives + 1e-8)
    recall_per_class = true_positives / (true_positives + false_negatives + 1e-8)

    f1_per_class = (
        2.0 * precision_per_class * recall_per_class
        / (precision_per_class + recall_per_class + 1e-8)
    )

    macro_f1 = float(f1_per_class.mean().item())
    return precision_per_class, recall_per_class, f1_per_class, macro_f1