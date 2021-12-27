from typing import Tuple
from sklearn import metrics

import torch

FALSE_POSITIVE_MASK = [0, 1]
FALSE_NEGATIVE_MASK = [1, 0]
TRUE_POSITIVE_MASK = [1, 1]
TRUE_NEGATIVE_MASK = [0, 0]


def _rowwise_select_by_mask(input_, mask):
    return torch.nonzero(
        torch.all(input_ == torch.tensor(mask, device=input_.device), dim=1)
    ).flatten()


def get_confusion_matrix_groups_indices(
    y_true: torch.Tensor, y_pred: torch.Tensor, threshold: float = 0.5
):
    """This method is responsible for returning the indices of corresponding groups classes
    (e.g. for false positives class it returns the indices on which all the false positives photos are located in the y_pred tensor)

    Args:
        y_true (torch.Tensor): Ground truths
        y_pred (torch.Tensor): Model predictions
        threshold (float, optional): Threshold separating positives from negatives. Defaults to 0.5.

    Returns:
        4x(torch.Tensor): Indices of classes positions in the y_pred tensor (where f.e. false positives items are in the y_pred tensor)
    """
    y_pred = (y_pred >= threshold).float()
    stacked_tensor = torch.stack((y_true, y_pred), dim=1)

    fp_indices = _rowwise_select_by_mask(stacked_tensor, FALSE_POSITIVE_MASK)
    fn_indices = _rowwise_select_by_mask(stacked_tensor, FALSE_NEGATIVE_MASK)
    tp_indices = _rowwise_select_by_mask(stacked_tensor, TRUE_POSITIVE_MASK)
    tn_indices = _rowwise_select_by_mask(stacked_tensor, TRUE_NEGATIVE_MASK)

    return fp_indices, fn_indices, tp_indices, tn_indices


def get_confusion_matrix_groups_sizes(
    y_true: torch.Tensor, y_pred: torch.Tensor, threshold: float = 0.5
) -> Tuple[int]:
    return tuple(
        map(len, get_confusion_matrix_groups_indices(y_true, y_pred, threshold))
    )


def accuracy(y_pred: torch.Tensor, y_true: torch.Tensor) -> float:
    #fp, fn, tp, tn = get_confusion_matrix_groups_sizes(y_true, y_pred, threshold)
    correct_pred = (y_pred == y_true).float()
    acc = correct_pred.sum() / len(correct_pred)
    #return (tp + tn) / (fp + fn + tp + tn)
    return acc


def cohen_kappa_score(y_pred: torch.Tensor, y_true: torch.Tensor) -> float:
    return metrics.cohen_kappa_score(
        y_pred.detach().cpu().numpy(),
        y_true.detach().cpu().numpy(),
        weights = "quadratic"
    )
