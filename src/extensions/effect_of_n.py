import csv
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# pylint: disable=wrong-import-position,import-error
from datasets import generate_synthetic_dataset
from models import RidgeRegression
from plot_results import BG, GRID, MUTED, RESULTS_DIR, TRAIN_COLOR, figure_title
from stability import leave_one_out_stability


# CONFIGURATION

N_VALUES = [25, 50, 100, 200, 400, 800, 1600, 3000]  # training set sizes (d = 20 --> 21 weights)
N_REPEATS = 5       # random subsets of the pool for each n
LOO_POINTS = 100    
NOISE_VARIANCE = 1.0  # best possible test MSE

# two ways of keeping the regularization fixed while n grows:
SETTINGS = {
    "alpha": [0.0, 1.0, 10.0, 100.0, 1000.0],  # 0 = least squares
    "lambda": [0.0, 0.1, 1.0, 10.0],
}


RAMP = ["#b79bd9", "#9266c0", "#6a3d9a", "#3f1f66"]


# EXPERIMENT

def alpha_for(setting, value, n):
    return value if setting == "alpha" else value * n


# one run: train on n random points of the pool, test on the fixed test set
def run_once(dataset, n, alpha, seed):
    idx = np.random.default_rng(seed).choice(dataset.n_train, size=n, replace=False)
    x_sub, y_sub = dataset.X_train[idx], dataset.y_train[idx]  # random subset of size n
    model = RidgeRegression(alpha).fit(x_sub, y_sub)
    stab = leave_one_out_stability(RidgeRegression, x_sub, y_sub, dataset.X_test, dataset.y_test,
                                   alpha, max_points=min(n, LOO_POINTS), random_state=seed)
    return (model.mse(x_sub, y_sub), model.mse(dataset.X_test, dataset.y_test),
            stab.mean_abs_pred_change)


def run_n_sweep(dataset):
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
    path.parent.mkdir(exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


# PLOT


def draw_lines(ax, rows, setting, column, label_fmt):
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


def style_axis(ax, title, subtitle, ylabel):
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.minorticks_off()
    ax.set_xlabel(r"$n$")
    ax.set_xticks(N_VALUES)  # one tick for each value of n used in the experiment
    ax.set_xticklabels([str(n) for n in N_VALUES])  # plain numbers, not powers of 10
    ax.set_ylabel(ylabel)
    ax.set_title(title, loc="left", fontsize=12, fontweight="bold", pad=24)
    ax.text(0, 1.03, subtitle, transform=ax.transAxes, fontsize=9, color=MUTED)
    ax.grid(axis="y", color=GRID, linewidth=1)
    ax.set_axisbelow(True)


def draw_one_over_n(ax, rows): # reference line with slope -1 (value proportional to 1/n)
    anchor = next(r["pred_change"] for r in rows
                  if r["setting"] == "alpha" and r["value"] == 0 and r["n"] == 200)
    ns = np.array(N_VALUES)
    ax.plot(ns, anchor * 200 / ns, color=MUTED, linestyle="--", linewidth=1, zorder=2,
            label=r"slope $-1$ (proportional to $1/n$)")


def legend_below(ax, extra_axis=None):
    
    handles, labels = ax.get_legend_handles_labels()
    if extra_axis is not None:
        for handle, label in zip(*extra_axis.get_legend_handles_labels()):
            if label not in labels:
                handles.append(handle)
                labels.append(label)
    ax.legend(handles, labels, frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.2),
              ncol=3, fontsize=8)


def plot_effect_of_n(rows):

    fig, axes = plt.subplots(2, 2, figsize=(11, 9.5), sharey="row")
    (stab_alpha, stab_lambda), (mse_alpha, mse_lambda) = axes

    draw_lines(stab_alpha, rows, "alpha", "pred_change", r"$\alpha = {:g}$")
    draw_one_over_n(stab_alpha, rows)
    style_axis(stab_alpha, r"Stability, fixed $\alpha$", r"$\alpha$ does not change with $n$",
               "mean |change in prediction|")

    draw_lines(stab_lambda, rows, "lambda", "pred_change", r"$\lambda = {:g}$")
    draw_one_over_n(stab_lambda, rows)
    style_axis(stab_lambda, r"Stability, fixed $\lambda$ ($\alpha=\lambda n$)",
               r"the penalty grows with $n$", "mean |change in prediction|")

    for ax, setting, title, subtitle in (
            (mse_alpha, "alpha", r"Test error, fixed $\alpha$", r"$\alpha$ does not change with $n$"),
            (mse_lambda, "lambda", r"Test error, fixed $\lambda$ ($\alpha=\lambda n$)",
             r"the penalty grows with $n$")):
        draw_lines(ax, rows, setting, "test_mse", r"$\{} = {{:g}}$".format(setting))
        ax.axhline(NOISE_VARIANCE, color=MUTED, linestyle=":", linewidth=1, zorder=2,
                   label="noise variance (best possible)")
        style_axis(ax, title, subtitle, "test MSE")

    # one legend per column
    legend_below(mse_alpha, extra_axis=stab_alpha)
    legend_below(mse_lambda, extra_axis=stab_lambda)

    figure_title(fig, "Effect of the training set size n",
                 "Synthetic data (d = 20), mean over 5 random training subsets for each n")
    fig.tight_layout(rect=(0, 0, 1, 0.92), h_pad=3.5)
    
    fig.savefig(RESULTS_DIR / "effect_of_n.png", dpi=200, bbox_inches="tight") # bbox_inches="tight" --> the legends below the panels are not cut off
    plt.close(fig)


# MAIN

def main():
    pool = generate_synthetic_dataset(n_samples=4000)  # 3000 training points + 1000 test points
    rows = run_n_sweep(pool)
    save_csv(rows, RESULTS_DIR / "n_sweep_synthetic.csv")
    plot_effect_of_n(rows)
    print(f"Saved results and figure in {RESULTS_DIR}")


if __name__ == "__main__":
    main()