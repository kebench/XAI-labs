from __future__ import annotations

from dataclasses import dataclass
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from xai_lab.core.metrics import confusion_matrix, precision_recall_f1_from_cm


@dataclass
class EvalResult:
    """
    Evaluation results for a model.
    """
    loss: float
    accuracy: float
    macro_f1: float
    cm: torch.Tensor


def train_one_epoch(model: nn.Module,
                    loader: DataLoader,
                    optimizer: torch.optim.Optimizer,
                    criterion: nn.Module,
                    device: torch.device) -> float:
    """
    One training epoch.

    Steps:
      - model.train(): enable dropout/bn updates
      - forward -> loss
      - backward -> optimizer.step()

    Returns average loss for the epoch.
    """
    model.train()
    total_loss = 0.0
    total_n = 0

    for x, y in loader:
        x = x.to(device)
        y = y.to(device)

        optimizer.zero_grad(set_to_none=True)
        logits = model(x)
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()

        bs = y.size(0)
        total_loss += loss.item() * bs
        total_n += bs

    return total_loss / max(total_n, 1)


@torch.no_grad()
def evaluate(model: nn.Module,
             loader: DataLoader,
             criterion: nn.Module,
             device: torch.device,
             num_classes: int) -> EvalResult:
    """
    Evaluation pass (validation or test).

    - model.eval(): disables dropout/bn updates
    - no_grad(): faster and avoids gradient memory

    Computes:
      - avg loss
      - accuracy
      - macro-F1
      - confusion matrix
    """
    model.eval()
    total_loss = 0.0
    total_n = 0

    all_pred = []
    all_true = []

    for x, y in loader:
        x = x.to(device)
        y = y.to(device)

        logits = model(x)
        loss = criterion(logits, y)

        pred = logits.argmax(dim=1)

        all_pred.append(pred.cpu())
        all_true.append(y.cpu())

        bs = y.size(0)
        total_loss += loss.item() * bs
        total_n += bs

    pred = torch.cat(all_pred)
    true = torch.cat(all_true)

    cm = confusion_matrix(pred, true, num_classes=num_classes)
    _, _, _, macro_f1 = precision_recall_f1_from_cm(cm)

    acc = (pred == true).to(torch.float32).mean().item()
    avg_loss = total_loss / max(total_n, 1)

    return EvalResult(loss=avg_loss, accuracy=acc, macro_f1=macro_f1, cm=cm)
