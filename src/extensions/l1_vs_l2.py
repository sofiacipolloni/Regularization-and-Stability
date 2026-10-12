"""
Extension 3: L1 (Lasso) against L2 (Ridge) regularization.
Lasso is written from scratch with coordinate descent. Both models are trained on
both datasets for a grid of regularization strengths. The two methods have
different alpha scales, so on the x axis we use the "size" of the weights
(L1 norm divided by the L1 norm of least squares: 1 = no shrinkage, 0 = all
weights are zero), which is comparable between Ridge and Lasso.
Results go to results/l1_vs_l2_<dataset>.csv and results/l1_vs_l2.png.
"""

import csv
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

# this file lives in src/extensions/ --> add src/ to the search path to import our own modules
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# pylint: disable=wrong-import-position,import-error
from datasets import generate_synthetic_dataset, load_real_dataset
from models import RidgeRegression
from plot_results import BG, GRID, MUTED, RESULTS_DIR, TEST_COLOR, figure_title
from stability import leave_one_out_stability


# CONFIGURATION

RIDGE_ALPHAS = [0.0] + [float(a) for a in np.logspace(-2, 6, 17)]  # 0 = least squares
# Lasso alpha = ratio * alpha_max, where alpha_max is the smallest alpha that sets ALL weights to 0
LASSO_RATIOS = [float(r) for r in np.logspace(-4, 0, 17)]
LOO_POINTS = 60       # leave-one-out on 60 random training points only --> keeps Lasso fast
ZERO_TOL = 1e-8       # a weight smaller than this counts as zero
LASSO_COLOR = "#2a9d8f"  # teal, to tell Lasso from Ridge (purple)
RIDGE_COLOR = TEST_COLOR
DATASET_TITLES = {"real_diabetes": "Real dataset (diabetes)", "synthetic": "Synthetic dataset"}


# LASSO FROM SCRATCH

# soft-thresholding: moves x towards 0 by t, and stops at exactly 0
def soft_threshold(x, t):
    """sign(x) * max(|x| - t, 0)."""
    return np.sign(x) * max(abs(x) - t, 0.0)


# column of ones in front of X --> the first weight is the intercept (same trick as models.py)
def add_bias_column(x):
    """Return [1 | x]."""
    return np.hstack([np.ones((x.shape[0], 1)), x])


class LassoRegression:
    """Minimizes ||y - S w||^2 + alpha * ||w||_1 (S = [1 | X]) with coordinate descent.

    Same convention as RidgeRegression: the intercept w[0] is penalized too (the data are
    already centered, so it is close to 0 anyway), which makes the two models comparable.
    """

    def __init__(self, alpha=1.0, max_iter=1000, tol=1e-9):
        self.alpha = alpha
        self.max_iter = max_iter
        self.tol = tol
        self.w = None

    def fit(self, x, y):
        """Update one weight at a time, keeping the others fixed, until nothing moves."""
        s = add_bias_column(x)
        col_sq = (s ** 2).sum(axis=0)           # ||S_j||^2 for every column j
        w = np.zeros(s.shape[1])
        residual = y.astype(float).copy()       # y - S w (w = 0 at the start)
        for _ in range(self.max_iter):
            biggest_change = 0.0
            for j in range(s.shape[1]):
                rho = s[:, j] @ residual + col_sq[j] * w[j]  # correlation without feature j
                new_wj = soft_threshold(rho, self.alpha / 2) / col_sq[j]
                residual += s[:, j] * (w[j] - new_wj)        # keep the residual up to date
                biggest_change = max(biggest_change, abs(new_wj - w[j]))
                w[j] = new_wj
            if biggest_change < self.tol:       # nothing moves anymore --> converged
                break
        self.w = w
        return self

    def predict(self, x):
        """Linear prediction S w."""
        return add_bias_column(x) @ self.w

    def mse(self, x, y):
        """Mean squared error."""
        return float(np.mean((y - self.predict(x)) ** 2))


# EXPERIMENT

# weights without the intercept (both models keep the intercept in w[0])
def weights(model):
    """The weights of the features only."""
    return model.w[1:]


# one row for one model trained on the full training set
def evaluate(model_factory, alpha, dataset, ls_l1_norm):
    """Test MSE, stability, size of the weights and number of non-zero weights."""
    model = model_factory(alpha).fit(dataset.X_train, dataset.y_train)
    stab = leave_one_out_stability(model_factory, dataset.X_train, dataset.y_train,
                                   dataset.X_test, dataset.y_test, alpha,
                                   max_points=LOO_POINTS)
    w = weights(model)
    return {
        "alpha": alpha,
        "l1_fraction": np.abs(w).sum() / ls_l1_norm,   # 1 = like least squares, 0 = all zero
        "nonzero": int(np.sum(np.abs(w) > ZERO_TOL)),
        "train_mse": model.mse(dataset.X_train, dataset.y_train),
        "test_mse": model.mse(dataset.X_test, dataset.y_test),
        "pred_change": stab.mean_abs_pred_change,
    }


# Ridge and Lasso sweeps on one dataset
def run_dataset(dataset):
    """Return the list of rows (dicts) for this dataset, Ridge first and then Lasso."""
    ls_model = RidgeRegression(0.0).fit(dataset.X_train, dataset.y_train)
    ls_l1_norm = np.abs(weights(ls_model)).sum()   # to measure the shrinkage of every model
    # smallest alpha for which every Lasso weight is exactly 0
    alpha_max = 2 * np.max(np.abs(add_bias_column(dataset.X_train).T @ dataset.y_train))
    rows = []
    for alpha in RIDGE_ALPHAS:
        rows.append({"method": "ridge", **evaluate(RidgeRegression, alpha, dataset, ls_l1_norm)})
    for ratio in LASSO_RATIOS:
        rows.append({"method": "lasso",
                     **evaluate(LassoRegression, ratio * alpha_max, dataset, ls_l1_norm)})
    return rows


# SAVING

def save_csv(rows, path):
    """Write a list of dicts to a CSV file."""
    path.parent.mkdir(exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


# PLOT

# one line per method for one column against the size of the weights
def draw_methods(ax, rows, column):
    """Ridge in purple, Lasso in teal; points are sorted by size of the weights."""
    for method, color in (("ridge", RIDGE_COLOR), ("lasso", LASSO_COLOR)):
        line = sorted((r for r in rows if r["method"] == method), key=lambda r: r["l1_fraction"])
        ax.plot([r["l1_fraction"] for r in line], [r[column] for r in line], color=color,
                linewidth=2, marker="o", markersize=5, markeredgecolor=BG,
                markeredgewidth=1.0, label=method.capitalize(), zorder=3)


# light grid, title + small subtitle (same look as plot_results.py)
def style_axis(ax, title, subtitle, ylabel, show_xlabel):
    """Apply the shared formatting to one panel."""
    ax.set_xlim(-0.03, 1.05)
    ax.set_ylabel(ylabel)
    if show_xlabel:
        ax.set_xlabel("size of the weights  (L1 norm / L1 norm of least squares)")
    ax.set_title(title, loc="left", fontsize=12, fontweight="bold", pad=24)
    ax.text(0, 1.03, subtitle, transform=ax.transAxes, fontsize=9, color=MUTED)
    ax.grid(axis="y", color=GRID, linewidth=1)
    ax.set_axisbelow(True)


def plot_l1_vs_l2(all_rows):
    """Rows = datasets; columns = test error, stability, number of non-zero weights."""
    fig, axes = plt.subplots(len(all_rows), 3, figsize=(15, 4.3 * len(all_rows)))
    columns = [("test_mse", "Test error", "test MSE"),
               ("pred_change", "Stability", "mean |change in prediction|"),
               ("nonzero", "Non-zero weights", "number of weights")]
    for i, (name, rows) in enumerate(all_rows.items()):
        for j, (column, title, ylabel) in enumerate(columns):
            draw_methods(axes[i, j], rows, column)
            subtitle = DATASET_TITLES[name] + ("  |  lower is better" if j < 2 else "")
            style_axis(axes[i, j], title, subtitle, ylabel, show_xlabel=i == len(all_rows) - 1)
        axes[i, 1].set_yscale("log")
    axes[0, 0].legend(frameon=False, loc="upper right", ncol=2, fontsize=9)
    figure_title(fig, "Ridge (L2) against Lasso (L1)",
                 "Same training data; each point is one value of alpha "
                 "(left = strong regularization)")
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(RESULTS_DIR / "l1_vs_l2.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


# MAIN

def main():
    """Run both methods on both datasets, save the tables and draw the figure."""
    all_rows = {}
    for dataset in (load_real_dataset(), generate_synthetic_dataset()):
        rows = run_dataset(dataset)
        save_csv(rows, RESULTS_DIR / f"l1_vs_l2_{dataset.name}.csv")
        all_rows[dataset.name] = rows
    plot_l1_vs_l2(all_rows)
    print(f"Saved results and figure in {RESULTS_DIR}")


if __name__ == "__main__":
    main()