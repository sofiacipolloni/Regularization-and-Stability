"""
Extension 2: effect of the number of features d on stability and test error.
Synthetic data only. The training set size is fixed (n = 200) while d grows
from 2 to 190, so the problem goes from "many more points than weights" to
"almost as many weights as points". For every d we compare least squares with a
few values of alpha, always on the same big test set, and average several
random datasets. Results go to results/d_sweep_synthetic.csv and
results/effect_of_d.png.
"""

import csv
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

# this file lives in src/extensions/ --> add src/ to the search path to import our own modules
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# pylint: disable=wrong-import-position,import-error
from datasets import generate_synthetic_dataset
from models import RidgeRegression
from plot_results import BG, GRID, MUTED, RESULTS_DIR, TRAIN_COLOR, figure_title
from stability import leave_one_out_stability


# CONFIGURATION

N_TRAIN = 200       # training set size, the same for every d
N_TEST = 1000       # test set size, big --> the test error is a precise estimate
D_VALUES = [2, 5, 10, 20, 50, 100, 150, 190]  # number of features (190 + intercept = 191 weights)
ALPHAS = [0.0, 1.0, 10.0, 100.0]  # 0 = least squares; alpha is the same for every d
N_REPEATS = 5       # random datasets (new X, new true weights, new noise) for each d
LOO_POINTS = 50     # leave-one-out on 50 random training points only --> keeps it fast
NOISE_VARIANCE = 1.0  # noise_std = 1 in generate_synthetic_dataset --> best possible test MSE

# purple ramp from weak to strong regularization (least squares is drawn in gold)
RAMP = ["#b79bd9", "#6a3d9a", "#3f1f66"]


# EXPERIMENT

# new synthetic dataset with d features: n_train = 200 training points, 1000 test points
def make_dataset(d, seed):
    """Dataset with d features; the seed changes X, the true weights and the noise."""
    return generate_synthetic_dataset(n_samples=N_TRAIN + N_TEST, n_features=d,
                                      test_size=N_TEST, random_state=seed)


# one run: train on the dataset, test on its test set
def run_once(dataset, alpha, seed):
    """Return train MSE, test MSE and mean |change in prediction| for this dataset."""
    model = RidgeRegression(alpha).fit(dataset.X_train, dataset.y_train)
    stab = leave_one_out_stability(RidgeRegression, dataset.X_train, dataset.y_train,
                                   dataset.X_test, dataset.y_test, alpha,
                                   max_points=LOO_POINTS, random_state=seed)
    return (model.mse(dataset.X_train, dataset.y_train),
            model.mse(dataset.X_test, dataset.y_test), stab.mean_abs_pred_change)


# every (d, alpha) combination, averaged over the random datasets
def run_d_sweep():
    """Return one row per (d, alpha) with the mean over the repeats."""
    rows = []
    for d in D_VALUES:
        datasets_d = [make_dataset(d, seed) for seed in range(N_REPEATS)]  # same for all alphas
        for alpha in ALPHAS:
            runs = np.array([run_once(ds, alpha, seed) for seed, ds in enumerate(datasets_d)])
            train, test, pred = runs.mean(axis=0)  # mean over the repeats
            rows.append({
                "d": d, "alpha": alpha,
                "train_mse": train, "test_mse": test, "gap": test - train,
                "pred_change": pred, "pred_change_std": runs[:, 2].std(),
            })
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

# curve of one column against d, one line per alpha (least squares in gold, the rest in purple)
def draw_lines(ax, rows, column):
    """Draw one line per alpha."""
    for i, alpha in enumerate(ALPHAS):
        line = [r for r in rows if r["alpha"] == alpha]
        color = TRAIN_COLOR if alpha == 0 else RAMP[i - 1]  # gold = least squares
        label = "least squares" if alpha == 0 else rf"$\alpha$ = {alpha:g}"
        ax.plot([r["d"] for r in line], [r[column] for r in line], color=color, linewidth=2,
                marker="o", markersize=6, markeredgecolor=BG, markeredgewidth=1.2,
                label=label, zorder=3)


# log y axis, light grid, title + small subtitle (same look as plot_results.py)
def style_axis(ax, title, subtitle, ylabel):
    """Apply the shared formatting to one panel."""
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.minorticks_off()
    ax.set_xticks([2, 5, 10, 20, 50, 100, 190])  # 150 is skipped, too close to 190 to read
    ax.set_xticklabels(["2", "5", "10", "20", "50", "100", "190"])
    ax.set_xlabel("number of features d")
    ax.set_ylabel(ylabel)
    ax.set_title(title, loc="left", fontsize=12, fontweight="bold", pad=24)
    ax.text(0, 1.03, subtitle, transform=ax.transAxes, fontsize=9, color=MUTED)
    ax.grid(axis="y", color=GRID, linewidth=1)
    ax.set_axisbelow(True)


# theory for least squares with Gaussian features: test MSE = sigma^2 * (1 + p / (n - p - 1))
def draw_theory(ax):
    """Dotted line: expected test MSE of least squares, p = d + 1 weights."""
    ds = np.array(D_VALUES)
    p = ds + 1
    ax.plot(ds, NOISE_VARIANCE * (1 + p / (N_TRAIN - p - 1)), color=MUTED, linestyle=":",
            linewidth=1.5, zorder=2, label="theory, least squares")


# legend under the panel, so it never covers the curves
def legend_below(ax):
    """Place the legend of this panel below the x-axis label."""
    ax.legend(frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=3, fontsize=8)


def plot_effect_of_d(rows):
    """Two panels: stability and test error against d."""
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.8))

    draw_lines(axes[0], rows, "pred_change")
    style_axis(axes[0], "Stability", "lower = more stable", "mean |change in prediction|")
    legend_below(axes[0])

    draw_lines(axes[1], rows, "test_mse")
    draw_theory(axes[1])
    axes[1].axhline(NOISE_VARIANCE, color=MUTED, linestyle="--", linewidth=1, zorder=2,
                    label="noise variance (best possible)")
    style_axis(axes[1], "Test error", "closer to the dashed line is better", "test MSE")
    legend_below(axes[1])

    figure_title(fig, "More features make the algorithm less stable",
                 "Synthetic data, 200 training points, mean over 5 random datasets for each d")
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    # bbox_inches="tight" --> the legends below the panels are not cut off
    fig.savefig(RESULTS_DIR / "effect_of_d.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


# MAIN

def main():
    """Run the sweep over d, save the table and draw the figure."""
    rows = run_d_sweep()
    save_csv(rows, RESULTS_DIR / "d_sweep_synthetic.csv")
    plot_effect_of_d(rows)
    print(f"Saved results and figure in {RESULTS_DIR}")


if __name__ == "__main__":
    main()