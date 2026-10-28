import csv

import numpy as np

from datasets import generate_synthetic_dataset, load_real_dataset
from models import RidgeRegression
from run_experiment import ALPHAS, RESULTS_DIR
from stability import leave_one_out_stability

N_FOLDS = 5


# K-FOLD SPLITTING
def make_folds(n_samples, n_folds, random_state=0):
    rng = np.random.default_rng(random_state) # random generator with fixed seed
    shuffled = rng.permutation(n_samples) # mix numbers 1 --> n-1 (indices of training points)
    return np.array_split(shuffled, n_folds)  # folds of almost equal size (66/67) --> validation set


# CROSS-VALIDATED ERROR OF ONE ALPHA
def cv_error(X_train, y_train, alpha, folds):
    errors = []
    for val_idx in folds: # each time: 1 fold as validation
        mask = np.ones(len(y_train), dtype=bool) # list of Trues (1 for each point)
        mask[val_idx] = False  # False = validation, True = used for training
        model = RidgeRegression(alpha).fit(X_train[mask], y_train[mask]) # training the model on k-1 folds
        errors.append(model.mse(X_train[val_idx], y_train[val_idx]))
    return float(np.mean(errors)), float(np.std(errors)) # error


# ALPHA SELECTION
def select_alpha(dataset, alphas, n_folds=N_FOLDS):
    folds = make_folds(dataset.n_train, n_folds)  # same folds for every alpha
    rows = []
    for alpha in alphas:
        mean, std = cv_error(dataset.X_train, dataset.y_train, alpha, folds)
        rows.append({"alpha": alpha, "cv_mse": mean, "cv_std": std})
    best = min(rows, key=lambda r: r["cv_mse"])  # compares rows looking at mse col --> lowest mean validation error
    return rows, best["alpha"]


# MAIN

def main():
    for ds in [load_real_dataset(), generate_synthetic_dataset()]:
        rows, best_alpha = select_alpha(ds, ALPHAS)

        chosen = RidgeRegression(best_alpha).fit(ds.X_train, ds.y_train) # model with chosen alpha trained on full training set
        ls = RidgeRegression(0.0).fit(ds.X_train, ds.y_train)  # least squares for comparison
        stab = leave_one_out_stability(RidgeRegression, ds.X_train, ds.y_train,
                                       ds.X_test, ds.y_test, best_alpha) # best alpha
        stab_ls = leave_one_out_stability(RidgeRegression, ds.X_train, ds.y_train,
                                          ds.X_test, ds.y_test, 0.0) # alpha = 0 

        # Table, best alpha, MSE errors, stabilities 
        print(f"\n=== {ds.name} ({N_FOLDS}-fold CV) ===")
        print(f"{'alpha':>10} {'cv_mse':>12} {'cv_std':>10}")
        for r in rows:
            print(f"{r['alpha']:>10.4g} {r['cv_mse']:>12.4f} {r['cv_std']:>10.4f}")
        print(f"best alpha (CV): {best_alpha:.4g}")
        print(f"test MSE: CV alpha = {chosen.mse(ds.X_test, ds.y_test):.4f} | "
              f"least squares = {ls.mse(ds.X_test, ds.y_test):.4f}") # MSE errors
        print(f"stability (pred change): CV alpha = {stab.mean_abs_pred_change:.5f} | "
              f"least squares = {stab_ls.mean_abs_pred_change:.5f}") # stabilities
        
        # create fold 
        RESULTS_DIR.mkdir(exist_ok=True)
        with open(RESULTS_DIR / f"cv_{ds.name}.csv", "w", newline="", encoding="utf-8") as f:
            #csv
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)


if __name__ == "__main__":
    main()