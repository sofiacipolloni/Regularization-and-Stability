import csv
from pathlib import Path

import numpy as np

from datasets import generate_synthetic_dataset, load_real_dataset
from models import RidgeRegression
from stability import leave_one_out_stability


# CONFIGURATION

# results/ folder next to src/ --> works wherever the script is launched from
RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"

# alpha = 0 is plain least squares; the other values are log-spaced, from
# almost no regularization (1e-3) to very strong regularization (1e4)
ALPHAS = [0.0] + [float(a) for a in np.logspace(-3, 4, 15)]  # float() --> clean values in the CSV


# ALPHA SWEEP

def run_alpha_sweep(dataset, alphas):
    """For each alpha: train/test MSE, test R2, generalization gap and stability."""  # R2
    rows = []  # one dict (= one CSV row) per alpha
    test_variance = np.var(dataset.y_test)  # variance of the test target --> same for every alpha  # R2
    for alpha in alphas:
        # model trained on the full training set
        model = RidgeRegression(alpha).fit(dataset.X_train, dataset.y_train)
        train_mse = model.mse(dataset.X_train, dataset.y_train)  # training error
        test_mse = model.mse(dataset.X_test, dataset.y_test)  # estimate of the true risk

        # full leave-one-out (no max_points), cheap on these small datasets
        stab = leave_one_out_stability(
            model_factory=RidgeRegression,  # calling it with alpha builds a fresh model
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
            "test_r2": 1 - test_mse / test_variance,  # share of test variance explained (1 = perfect, 0 = predicts the mean)  # R2
            "gap": test_mse - train_mse,  # empirical proxy of R - R_emp (Theorem 12)
            "pred_change": stab.mean_abs_pred_change,  # main stability measure
            "loss_change": stab.mean_abs_loss_change,
        })
    return rows


# SAVING AND PRINTING

# save the sweep results to disk --> the plotting script will read the CSV
# without recomputing anything, and the numbers stay available for the report
def save_csv(rows, path):
    """Write a list of dicts to a CSV file."""
    path.parent.mkdir(exist_ok=True)  # create results/ if it does not exist
    with open(path, "w", newline="", encoding="utf-8") as f:  # newline="" avoids blank lines between rows
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))  # columns = dict keys
        writer.writeheader()  # first line of the file: column names
        writer.writerows(rows)  # one line per alpha


# show the same results in the terminal as an aligned table, to check
# them at a glance while the script runs
def print_table(name, rows):
    """Print the sweep results as a readable table."""
    print(f"\n=== {name} ===")  # dataset title, preceded by an empty line
    # header: ">10" = right-aligned in a column 10 characters wide
    print(f"{'alpha':>10} {'train_mse':>12} {'test_mse':>12} {'test_r2':>9} {'gap':>10} "  # R2
          f"{'pred_chg':>10} {'loss_chg':>10}")
    for r in rows:  # one line per alpha
        # ".4g" = 4 significant digits (compact for alpha: 0.001, 100, 1e+04);
        # ".4f" / ".5f" = fixed number of decimals
        print(f"{r['alpha']:>10.4g} {r['train_mse']:>12.4f} {r['test_mse']:>12.4f} "
              f"{r['test_r2']:>9.4f} {r['gap']:>10.4f} "  # R2
              f"{r['pred_change']:>10.5f} {r['loss_change']:>10.4f}")


# MAIN

def main():
    """Run the alpha sweep on both datasets, print and save the results."""
    all_datasets = [load_real_dataset(), generate_synthetic_dataset()]
    for ds in all_datasets:
        rows = run_alpha_sweep(ds, ALPHAS)
        print_table(ds.name, rows)
        save_csv(rows, RESULTS_DIR / f"sweep_{ds.name}.csv")  # one CSV per dataset


if __name__ == "__main__":  # run main() only if the file is launched directly
    main()