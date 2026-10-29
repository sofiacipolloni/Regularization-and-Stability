import csv
from pathlib import Path
import numpy as np
from datasets import generate_synthetic_dataset, load_real_dataset
from models import RidgeRegression
from stability import leave_one_out_stability


# CONFIGURATION
RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"

# Choice of alpha values 
ALPHAS = [0.0] + [float(a) for a in np.logspace(-3, 4, 15)]  # list comprehension
# alpha = 0: least squares + log-spaced alphas (almost no regularization --> very strong regularization)



# ALPHA SWEEP
def run_alpha_sweep(dataset, alphas):
    """For each alpha: train/test MSE, test R2, generalization gap and stability."""
    rows = []  # one dict (= one CSV row) for each alpha
    test_variance = np.var(dataset.y_test)  # does not depend on alpha, so computed once
    for alpha in alphas:
        model = RidgeRegression(alpha).fit(dataset.X_train, dataset.y_train)  # compute the weights
        train_mse = model.mse(dataset.X_train, dataset.y_train)
        test_mse = model.mse(dataset.X_test, dataset.y_test)  # estimate of the true risk

        # full leave-one-out (no max_points): fast enough with small datasets
        stab = leave_one_out_stability(
            model_factory=RidgeRegression,  # called with alpha, it builds a new model
            X_train=dataset.X_train,
            y_train=dataset.y_train,
            X_test=dataset.X_test,
            y_test=dataset.y_test,
            alpha=alpha,
        )

        rows.append({
            "alpha": alpha,
            "train_mse": train_mse,
            "test_mse": test_mse,
            "test_r2": 1 - test_mse / test_variance,  # fraction of test variance explained
            "gap": test_mse - train_mse,  # empirical version of R - R_emp
            "pred_change": stab.mean_abs_pred_change,  # main stability measure
            "loss_change": stab.mean_abs_loss_change,
        })
    return rows


# SAVING AND PRINTING
def save_csv(rows, path):
    path.parent.mkdir(exist_ok=True)  # create the results folder if it is not there yet
    with open(path, "w", newline="", encoding="utf-8") as f:  # "w" overwrites the old file
        
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))  # columns = keys of a row
        writer.writeheader()  # first line: column names
        writer.writerows(rows)  # one line for each alpha


def print_table(name, rows): # print the rows as a table

    print(f"\n=== {name} ===") # header: names right-aligned 
    print(f"{'alpha':>10} {'train_mse':>12} {'test_mse':>12} {'test_r2':>9} {'gap':>10} "
          f"{'pred_chg':>10} {'loss_chg':>10}")
    for r in rows:
        print(f"{r['alpha']:>10.4g} {r['train_mse']:>12.4f} {r['test_mse']:>12.4f} "
              f"{r['test_r2']:>9.4f} {r['gap']:>10.4f} "
              f"{r['pred_change']:>10.5f} {r['loss_change']:>10.4f}")
              # ".4g" = 4 significant digits, ".4f" = 4 decimals

# MAIN
def main():
    all_datasets = [load_real_dataset(), generate_synthetic_dataset()]
    for ds in all_datasets:
        rows = run_alpha_sweep(ds, ALPHAS)
        print_table(ds.name, rows)
        save_csv(rows, RESULTS_DIR / f"sweep_{ds.name}.csv")  # one CSV for each dataset


if __name__ == "__main__":  # run main() only when the file is launched directly
    main()