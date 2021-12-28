from sklearn import metrics
import torch


def accuracy(y_pred: torch.Tensor, y_true: torch.Tensor) -> float:
    correct_pred = (y_pred == y_true).float()
    acc = correct_pred.sum() / len(correct_pred)
    return acc


def cohen_kappa_score(y_pred: torch.Tensor, y_true: torch.Tensor) -> float:
    return metrics.cohen_kappa_score(
        y_pred.detach().cpu().numpy(),
        y_true.detach().cpu().numpy(),
        weights = "quadratic"
    )
