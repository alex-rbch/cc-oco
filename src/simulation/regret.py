from __future__ import annotations

import numpy as np


def linear_regret(
    played_actions: np.ndarray, gradients: np.ndarray, radius: float
) -> float:
    if played_actions.shape != gradients.shape:
        raise ValueError("actions and gradients must have the same shape")
    learner_loss = float(np.sum(played_actions * gradients))
    return learner_loss + radius * float(np.linalg.norm(gradients.sum(axis=0)))
