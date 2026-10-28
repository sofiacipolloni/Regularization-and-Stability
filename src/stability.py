from dataclasses import dataclass
from typing import Callable

import numpy as np



# RESULT CONTAINER

@dataclass
class StabilityResult:
    """Aggregated output of a leave-one-out stability estimation."""

    alpha: float
    mean_abs_pred_change: float  # avg change in predictions (main stability measure)
    mean_abs_loss_change: float  # avg change in the squared loss
    per_point_pred_change: np.ndarray  # one value per removed training point



# LEAVE-ONE-OUT STABILITY

def leave_one_out_stability( 
    model_factory: Callable[[float], object],
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    alpha: float,
    max_points: int | None = None, 
    random_state: int = 0,
) -> StabilityResult: # gives back a new model
    
    n_train = X_train.shape[0] # n. training points

    reference_model = model_factory(alpha) # trained on the full training set --> the function works with any model
    reference_model.fit(X_train, y_train) 
    reference_preds = reference_model.predict(X_test) # predictions
    reference_losses = (y_test - reference_preds) ** 2 # losses

    if max_points is not None and max_points < n_train: # if max_points is set --> random subset of points removed
        rng = np.random.default_rng(random_state) # same random sample always
        indices_to_remove = rng.choice(n_train, size=max_points, replace=False)
    else: # if max_points is not set -->  every point is removed once
        indices_to_remove = np.arange(n_train)

    #lists 
    pred_changes = [] 
    loss_changes = []

    # cycle: 
    for i in indices_to_remove:
        # training set without point i
        X_loo = np.delete(X_train, i, axis=0)
        y_loo = np.delete(y_train, i, axis=0)

        # Retrain with the same alpha --> difference between two models is the missing point
        loo_model = model_factory(alpha)
        loo_model.fit(X_loo, y_loo)

        # Average over the test points of how much prediction / loss moved because of removing point i
        loo_preds = loo_model.predict(X_test)
        loo_losses = (y_test - loo_preds) ** 2

        pred_changes.append(np.mean(np.abs(loo_preds - reference_preds)))
        loss_changes.append(np.mean(np.abs(loo_losses - reference_losses)))

    pred_changes = np.array(pred_changes)
    loss_changes = np.array(loss_changes)

    return StabilityResult(
        alpha=alpha,
        mean_abs_pred_change=float(pred_changes.mean()),
        mean_abs_loss_change=float(loss_changes.mean()),
        per_point_pred_change=pred_changes,
    )