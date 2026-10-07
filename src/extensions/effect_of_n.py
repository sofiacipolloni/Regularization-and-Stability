"""
Extension 1: effect of the training set size n on stability and generalization gap.
Synthetic data only (we can generate as many points as we need). For each n we
train on a random subset of a big training pool, always testing on the same
test set, and repeat several times to average out the randomness of the subset.
Results go to results/n_sweep_synthetic.csv and results/effect_of_n.png.
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

N_VALUES = [25, 50, 100, 200, 400, 800, 1600, 3000]  # training set sizes (d = 20 --> 21 weights)
N_REPEATS = 5       # random subsets of the pool for each n
LOO_POINTS = 100    # leave-one-out on 100 random points only --> keeps the big n fast
NOISE_VARIANCE = 1.0  # noise_std = 1 in generate_synthetic_dataset --> best possible test MSE

# two ways of keeping the regularization "the same" while n grows:
#   "alpha"  = same alpha for every n (the penalty gets relatively weaker as n grows)
#   "lambda" = alpha = lambda * n, i.e. the penalty grows with n (lambda = alpha / n)
SETTINGS = {
    "alpha": [0.0, 1.0, 10.0, 100.0, 1000.0],  # 0 = least squares
    "lambda": [0.0, 0.1, 1.0, 10.0],
}

# purple ramp from weak to strong regularization (least squares is drawn in gold)
RAMP = ["#b79bd9", "#9266c0", "#6a3d9a", "#3f1f66"]


# EXPERIMENT

# alpha actually used for a given setting, value and n
def alpha_for(setting, value, n):
    """alpha itself for 'alpha', value * n for 'lambda'."""
    return value if setting == "alpha" else value * n


# one run: train on n random points of the pool, test on the fixed test set
def run_once(dataset, n, alpha, seed):
    """Return train MSE, test MSE and mean |change in prediction| for this subset."""
    idx = np.random.default_rng(seed).choice(dataset.n_train, size=n, replace=False)
    x_sub, y_sub = dataset.X_train[idx], dataset.y_train[idx]  # random subset of size n
    model = RidgeRegression(alpha).fit(x_sub, y_sub)
    stab = leave_one_out_stability(RidgeRegression, x_sub, y_sub, dataset.X_test, dataset.y_test,
                                   alpha, max_points=min(n, LOO_POINTS), random_state=seed)
    return (model.mse(x_sub, y_sub), model.mse(dataset.X_test, dataset.y_test),
            stab.mean_abs_pred_change)


# every (setting, value, n) combination, averaged over the repeats
def run_n_sweep(dataset):
    """Return one row per (setting, value, n) with the mean over the repeats."""
    rows = []
    for setting, values in SETTINGS.items():
        for value in values:
            for n in N_VALUES:
                alpha = alpha_for(setting, value, n)
                runs = np.array([run_once(dataset, n, alpha, seed) for seed in range(N_REPEATS)])
                train, test, pred = runs.mean(axis=0)  # mean over the repeats
                rows.append({
                    "setting": setting, "value": value, "n": n,
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

# curve of one column against n for one setting, one line per value
def draw_lines(ax, rows, setting, column, label_fmt):
    """Draw one line per value of the setting; least squares in gold, the rest in purple."""
    values = SETTINGS[setting]
    shades = RAMP[len(RAMP) - len(values) + 1:]  # as many purples as non-zero values
    for i, value in enumerate(values):
        line = [r for r in rows if r["setting"] == setting and r["value"] == value]
        ns = [r["n"] for r in line]
        ys = [r[column] for r in line]
        color = TRAIN_COLOR if value == 0 else shades[i - 1]  # gold = least squares
        label = "least squares" if value == 0 else label_fmt.format(value)
        ax.plot(ns, ys, color=color, linewidth=2, marker="o", markersize=6,
                markeredgecolor=BG, markeredgewidth=1.2, label=label, zorder=3)


# log-log axes, light grid, title + small subtitle (same look as plot_results.py)
def style_axis(ax, title, subtitle, ylabel):
    """Apply the shared formatting to one panel."""
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.minorticks_off()
    ax.set_xlabel("training set size n")
    ax.set_ylabel(ylabel)
    ax.set_title(title, loc="left", fontsize=12, fontweight="bold", pad=24)
    ax.text(0, 1.03, subtitle, transform=ax.transAxes, fontsize=9, color=MUTED)
    ax.grid(axis="y", color=GRID, linewidth=1)
    ax.set_axisbelow(True)


# dashed reference line with slope -1 (value proportional to 1/n), through one point
def draw_one_over_n(ax, rows):
    """Reference line ~ 1/n, anchored on least squares at n = 200."""
    anchor = next(r["pred_change"] for r in rows
                  if r["setting"] == "alpha" and r["value"] == 0 and r["n"] == 200)
    ns = np.array(N_VALUES)
    ax.plot(ns, anchor * 200 / ns, color=MUTED, linestyle="--", linewidth=1, zorder=2,
            label="slope -1 (1/n)")


# legend under the panel, so it never covers the curves
def legend_below(ax):
    """Place the legend of this panel below the x-axis label."""
    ax.legend(frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=3, fontsize=8)


def plot_effect_of_n(rows):
    """Three panels: stability (fixed alpha), stability (alpha = lambda * n), test error."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.8))

    draw_lines(axes[0], rows, "alpha", "pred_change", r"$\alpha$ = {:g}")
    draw_one_over_n(axes[0], rows)
    style_axis(axes[0], "Stability, same alpha", "alpha does not change with n",
               "mean |change in prediction|")
    legend_below(axes[0])

    draw_lines(axes[1], rows, "lambda", "pred_change", r"$\alpha = {:g}\,n$")
    draw_one_over_n(axes[1], rows)
    style_axis(axes[1], "Stability, alpha = lambda * n", "the penalty grows with n",
               "mean |change in prediction|")
    legend_below(axes[1])

    draw_lines(axes[2], rows, "alpha", "test_mse", r"$\alpha$ = {:g}")
    axes[2].axhline(NOISE_VARIANCE, color=MUTED, linestyle=":", linewidth=1, zorder=2,
                    label="noise variance (best possible)")
    style_axis(axes[2], "Test error, same alpha", "closer to the dotted line is better", "test MSE")
    legend_below(axes[2])

    figure_title(fig, "More data makes the algorithm more stable",
                 "Synthetic data (d = 20), mean over 5 random training subsets for each n")
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    # bbox_inches="tight" --> the legends below the panels are not cut off
    fig.savefig(RESULTS_DIR / "effect_of_n.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


# MAIN

def main():
    """Run the sweep over n, save the table and draw the figure."""
    pool = generate_synthetic_dataset(n_samples=4000)  # 3000 training points + 1000 test points
    rows = run_n_sweep(pool)
    save_csv(rows, RESULTS_DIR / "n_sweep_synthetic.csv")
    plot_effect_of_n(rows)
    print(f"Saved results and figure in {RESULTS_DIR}")


if __name__ == "__main__":
    main()