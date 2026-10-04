"""Class-weighted cross-entropy and temperature scaling."""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
from sklearn.utils.class_weight import compute_class_weight


def balanced_class_weights(train_labels, labels: list[str]) -> np.ndarray:
    """sklearn 'balanced' weights, n / (n_classes * count), from the TRAINING folds only."""
    return compute_class_weight("balanced", classes=np.arange(len(labels)),
                                y=[labels.index(c) for c in train_labels])


def weighted_cross_entropy(weights: np.ndarray) -> nn.CrossEntropyLoss:
    return nn.CrossEntropyLoss(weight=torch.tensor(weights, dtype=torch.float32))


def fit_temperature(logits: np.ndarray, y: np.ndarray) -> float:
    """Single-parameter temperature scaling (Guo et al. 2017), fitted on VALIDATION logits."""
    logits_t, y_t = torch.tensor(logits, dtype=torch.float32), torch.tensor(y, dtype=torch.long)
    log_t = torch.zeros(1, requires_grad=True)
    optimiser = torch.optim.LBFGS([log_t], lr=0.1, max_iter=200)

    def closure():
        optimiser.zero_grad()
        loss = nn.functional.cross_entropy(logits_t / log_t.exp(), y_t)
        loss.backward()
        return loss

    optimiser.step(closure)
    return round(float(log_t.exp()), 4)
