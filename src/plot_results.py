"""
Plots for the alpha sweep: reads the CSV files written by run_experiment.py
(sweep) and cross_validation.py (alpha chosen by CV) and draws the two figures
required by the assignment (stability vs alpha, test error vs alpha).
Nothing is recomputed here: run those two scripts first.
"""

import csv
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"


DATASETS = [
    ("real_diabetes", "Real dataset (diabetes)"),
    ("synthetic", "Synthetic dataset"),
] #name of file, title


# LOOK

# line colors
TEST_COLOR = "#6a3d9a"   # plum: test
TRAIN_COLOR = "#c47f00"  # gold: training series

# line styles
LINE_STYLE = {"linestyle": "-", "marker": "o", "markersize": 6} # solid line with dots

# neutral colors
BG = "#fbf8f3"     # background
INK = "#2b2622"    # titles, markers and main text
MUTED = "#79726a"  # secondary text and reference lines
GRID = "#e9e3d6"   # grid lines and axes

# fonts: the first one installed on your computer is used (the last one always exists)
FONTS = ["Avenir Next", "Futura", "Helvetica Neue", "Arial", "DejaVu Sans"]

# global style
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": FONTS,
    "font.size": 10,
    "text.color": INK,
    "axes.labelcolor": MUTED,
    "axes.edgecolor": GRID,
    "xtick.color": MUTED,
    "ytick.color": MUTED,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "figure.facecolor": BG,
    "axes.facecolor": BG,
    "savefig.facecolor": BG,
}) 



# LOADING

# read one sweep CSV --> dict {column name: list of floats}
def load_sweep(name):
    """Load results/sweep_<name>.csv as a dict of float lists."""
    with open(RESULTS_DIR / f"sweep_{name}.csv", newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    return {key: [float(r[key]) for r in rows] for key in rows[0]}


# alpha = 0 cannot be drawn on a log axis --> it is returned separately
# (as the "least squares" reference) and removed from the curve
def split_least_squares(data, key):
    """Return (alphas > 0, values for alphas > 0, value at alpha = 0)."""
    alphas = [a for a in data["alpha"] if a > 0]
    values = [v for a, v in zip(data["alpha"], data[key]) if a > 0]
    ls_value = data[key][data["alpha"].index(0.0)]
    return alphas, values, ls_value

# CV alpha -> training set used to pick it
def load_cv_alpha(name):
    """Return the alpha selected by cross-validation for this dataset."""
    with open(RESULTS_DIR / f"cv_{name}.csv", newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    best = min(rows, key=lambda r: float(r["cv_mse"])) # keep row with lowest mse (alpha selected by cv)
    return float(best["alpha"])


# test error alpha --> test set used to pick it (keep for comparison with CV alpha)
def test_min_alpha(alphas, test):
    """Return the alpha with the lowest test MSE."""
    return alphas[test.index(min(test))]


# STYLE

# Diamonds (alpha on graph)
def highlight_style(kind):
    """Matplotlib marker settings for the 'cv' (filled) or 'test' (hollow) diamond."""
    if kind == "cv":
        return {"marker": "D", "markersize": 11, "markerfacecolor": INK,
                "markeredgecolor": BG, "markeredgewidth": 1.0} # cv alpha
    return {"marker": "D", "markersize": 11, "markerfacecolor": "none",
            "markeredgecolor": INK, "markeredgewidth": 1.8} #min alpha


def mark_alpha(ax, alphas, values, alpha, kind):
    """Highlight the point of the curve at this alpha."""
    i = alphas.index(alpha) # receives alpha
    ax.plot(alphas[i], values[i], linestyle="none", zorder=5, **highlight_style(kind)) # puts diamond on the stability value of that alpha


# Curve
def draw_curve(ax, alphas, values, series):
    """Draw one series as a 2px line, colored according to the series."""
    color = TEST_COLOR if series == "test" else TRAIN_COLOR # chooses color based on series
    ax.plot(alphas, values, color=color, linewidth=2, markeredgecolor=BG,
            markeredgewidth=1.2, zorder=3, **LINE_STYLE) # background color --> dots readable when they overlap a line


# Legend
def series_handle(series, label):
    """Legend entry for the 'test' or 'train' series."""
    color = TEST_COLOR if series == "test" else TRAIN_COLOR
    return Line2D([], [], color=color, linewidth=2, markeredgecolor=BG, label=label,
                  **LINE_STYLE)


# Panels' look
def style_axis(ax, title, subtitle, ylabel):
    """Apply the shared formatting to one panel."""
    ax.set_xscale("log")
    ax.minorticks_off()  # fewer ticks --> cleaner axis
    ax.set_xlabel(r"regularization strength $\alpha$")
    ax.set_ylabel(ylabel)
    ax.set_title(title, loc="left", fontsize=12, fontweight="bold", pad=24)
    ax.text(0, 1.03, subtitle, transform=ax.transAxes, fontsize=9, color=MUTED)
    ax.grid(axis="y", color=GRID, linewidth=1)
    ax.set_axisbelow(True)  # grid behind the data


# Subtitles
def alpha_subtitle(cv_alpha, best_test_alpha):
    """Text with the alpha chosen by CV and the alpha with the lowest test MSE."""
    return (rf"CV: $\alpha \approx {cv_alpha:.3g}$   |   "
            rf"lowest test MSE: $\alpha \approx {best_test_alpha:.3g}$")


# Legend voice for diamonds
def marker_handles():
    """Legend entries that explain the two highlight markers."""
    return [
        Line2D([], [], linestyle="none", label=r"$\alpha$ chosen by CV",
               **highlight_style("cv")),
        Line2D([], [], linestyle="none", label=r"$\alpha$ chosen by lowest test MSE",
               **highlight_style("test")),
    ]


# Shared legend
def figure_legend(fig, handles, ncol):
    """Place a shared legend at the bottom of the figure."""
    fig.legend(handles=handles, loc="lower center", ncol=ncol, frameon=False,
               fontsize=9, labelcolor=INK, columnspacing=2, handletextpad=0.6)


# Global title + subtitle  
def figure_title(fig, title, subtitle):
    """Write the figure title and a gray description under it."""
    fig.suptitle(title, x=0.015, y=0.99, ha="left", fontsize=15, fontweight="bold")
    fig.text(0.015, 0.905, subtitle, ha="left", fontsize=10, color=MUTED)
    
    
    
    
# FIGURE 1: STABILITY VS ALPHA

# same figure for two stability metrics: change in prediction (default) or change in loss
def plot_stability(column="pred_change", what="prediction",
                   filename="stability_vs_alpha.png", label="Stability"):
    """Mean |change in <what>| after removing one training point."""
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.8))  # 1 row, 2 panels (different scales)
    for ax, (name, title) in zip(axes, DATASETS):
        data = load_sweep(name)
        alphas, values, ls_value = split_least_squares(data, column)  # column of the CSV to plot
        _, test, _ = split_least_squares(data, "test_mse")
        cv_alpha = load_cv_alpha(name)
        best_test_alpha = test_min_alpha(alphas, test)

        ax.fill_between(alphas, values, color=TEST_COLOR, alpha=0.08)  # area under the curve
        draw_curve(ax, alphas, values, "test")
        mark_alpha(ax, alphas, values, cv_alpha, "cv") # filled
        mark_alpha(ax, alphas, values, best_test_alpha, "test") # empty

        ax.axhline(ls_value, color=MUTED, linestyle="--", linewidth=1) # dashed line = least squares (alpha = 0)
        ax.annotate(r"least squares ($\alpha=0$)", xy=(alphas[-1], ls_value),
                    xytext=(0, 5), textcoords="offset points", ha="right",
                    fontsize=8, color=MUTED)
      
        # headline number: how much more stable at the strongest regularization
        drop = 100 * (1 - values[-1] / ls_value)  # 1 - (last value, alpha = 10000) / (least squares, alpha = 0)
        ax.annotate(f"-{drop:.0f}% vs least squares", xy=(alphas[-1], values[-1]),
                    xytext=(-12, -3), textcoords="offset points", ha="right", va="center",
                    fontsize=9, color=INK, fontweight="bold")  # left of the last point

        style_axis(ax, title, alpha_subtitle(cv_alpha, best_test_alpha),
                   f"mean |change in {what}|")
        ax.set_ylim(bottom=0)  # y-axis starts from 0
    figure_title(fig, "Stability increases as regularization increases",
                 f"Average change in the test {what} when one training point "
                 "is removed (leave-one-out): lower = more stable")
    curve_handle = series_handle("test", label)
    ls_handle = Line2D([], [], color=MUTED, linestyle="--", linewidth=1,
                       label=r"Least squares ($\alpha=0$)")
    figure_legend(fig, [curve_handle, ls_handle] + marker_handles(), ncol=4)
    fig.tight_layout(rect=(0, 0.07, 1, 0.95))  # leave room for title above and legend below
    fig.savefig(RESULTS_DIR / filename, dpi=200)  # file name comes from the parameter
    plt.close(fig)



# FIGURE 2: TEST ERROR VS ALPHA

def plot_test_error():
    """Train and test MSE as a function of alpha (bias-variance trade-off)."""
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.8))
    for ax, (name, title) in zip(axes, DATASETS):
        data = load_sweep(name)
        alphas, test, test_ls = split_least_squares(data, "test_mse")
        _, train, _ = split_least_squares(data, "train_mse")
        cv_alpha = load_cv_alpha(name)
        best_test_alpha = test_min_alpha(alphas, test)

        # gray band = generalization gap (test error above training error)
        ax.fill_between(alphas, train, test, where=[te >= tr for te, tr in zip(test, train)],
                        color=MUTED, alpha=0.15, linewidth=0)
        draw_curve(ax, alphas, test, "test")
        draw_curve(ax, alphas, train, "train")
        # dashed line = test error of least squares (alpha = 0)
        ax.axhline(test_ls, color=MUTED, linestyle="--", linewidth=1)

        # filled diamond = alpha chosen by cross-validation (training set only),
        # hollow diamond = alpha with the lowest test error (read on the test curve)
        mark_alpha(ax, alphas, test, cv_alpha, "cv")
        mark_alpha(ax, alphas, test, best_test_alpha, "test")
        style_axis(ax, title, alpha_subtitle(cv_alpha, best_test_alpha), "mean squared error")
    figure_title(fig, "Training and test error vs regularization",
                 "Mean squared error on the training set and on the held-out test set")
    handles = [
        series_handle("test", "test MSE"),
        series_handle("train", "train MSE"),
        Patch(facecolor=MUTED, alpha=0.15, label="generalization gap (test - train)"),
        Line2D([], [], color=MUTED, linestyle="--", linewidth=1,
               label=r"test MSE, least squares ($\alpha=0$)"),
    ]
    figure_legend(fig, handles + marker_handles(), ncol=3)
    fig.tight_layout(rect=(0, 0.1, 1, 0.95))
    fig.savefig(RESULTS_DIR / "test_error_vs_alpha.png", dpi=200)
    plt.close(fig)


if __name__ == "__main__":
    plot_stability()  # change in prediction --> stability_vs_alpha.png
    plot_stability("loss_change", "loss", "stability_loss_vs_alpha.png", "Change in loss")
    plot_test_error()
    print(f"Figures saved in {RESULTS_DIR}")