import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)

from .data import DEVICE


def calculate_metrics(y_true, y_prob):
    y_true, y_prob = np.array(y_true), np.array(y_prob)
    y_pred = (y_prob > 0.5).astype(int)
    if len(np.unique(y_true)) < 2:
        return 0.5, 0.0, 0.5, 0.0, 0.0, 0.0, 0.0
    return (
        roc_auc_score(y_true, y_prob),
        average_precision_score(y_true, y_prob),
        accuracy_score(y_true, y_pred),
        f1_score(y_true, y_pred),
        matthews_corrcoef(y_true, y_pred),
        precision_score(y_true, y_pred, zero_division=0),
        recall_score(y_true, y_pred, zero_division=0),
    )


@torch.no_grad()
def evaluate_model(model, loader, target_etype=("medicine", "has_reaction", "reaction")):
    model.eval()
    y_true, y_prob = [], []
    for batch in loader:
        batch = batch.to(DEVICE)
        y_prob.extend(torch.sigmoid(model(batch)).cpu().numpy())
        y_true.extend(batch[target_etype].edge_label.cpu().numpy())
    return calculate_metrics(y_true, y_prob)

