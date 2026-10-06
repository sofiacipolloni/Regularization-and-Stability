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

# datasets to plot: (name used in the CSV file, title shown on the panel)
DATASETS = [
    ("real_diabetes", "Real dataset (diabetes)"),
    ("synthetic", "Synthetic dataset"),
]

# LOOK

# line colors: (test / main series, training series)
# options: "plum_gold", "indigo_raspberry", "burgundy_teal", "plum_teal"
COLORS = "plum_gold"

# line style: "dots" (solid lines with dots), "clean" (solid lines, no dots),
# "dashed" (training line dashed with square dots, easy to tell apart even in black and white)
LINES = "dots"

PALETTES = {
    "plum_gold": ("#6a3d9a", "#c47f00"),
    "indigo_raspberry": ("#4338ca", "#e11d74"),
    "burgundy_teal": ("#9b2c5a", "#2a9d8f"),
    "plum_teal": ("#6a3d9a", "#2a9d8f"),
}

LINE_STYLES = {
    "dots": {"test": {"linestyle": "-", "marker": "o", "markersize": 6},
             "train": {"linestyle": "-", "marker": "o", "markersize": 6}},
    "clean": {"test": {"linestyle": "-", "marker": "none"},
              "train": {"linestyle": "-", "marker": "none"}},
    "dashed": {"test": {"linestyle": "-", "marker": "o", "markersize": 6},
               "train": {"linestyle": "--", "marker": "s", "markersize": 5}},
}

TEST_COLOR, TRAIN_COLOR = PALETTES[COLORS]

# neutral colors (warm sand background)
BG = "#fbf8f3"     # background
INK = "#2b2622"    # titles, markers and main text
MUTED = "#79726a"  # secondary text and reference lines
GRID = "#e9e3d6"   # grid lines and axes

# fonts: the first one installed on your computer is used (the last one always exists)
FONTS = ["Avenir Next", "Futura", "Helvetica Neue", "Arial", "DejaVu Sans"]

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


# alpha chosen by cross-validation = the one with the lowest mean validation
# error in results/cv_<name>.csv (written by cross_validation.py)
def load_cv_alpha(name):
    """Return the alpha selected by cross-validation for this dataset."""
    with open(RESULTS_DIR / f"cv_{name}.csv", newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    best = min(rows, key=lambda r: float(r["cv_mse"]))
    return float(best["alpha"])


# alpha with the lowest test error, read directly on the test curve
# --> optimistic: the test set was used to pick it (kept only for comparison)
def test_min_alpha(alphas, test):
    """Return the alpha with the lowest test MSE."""
    return alphas[test.index(min(test))]


# STYLE

# look of the two highlight markers (both diamonds)
# filled diamond = alpha chosen by CV (the official choice, uses the training set only)
# hollow diamond = alpha with the lowest test error (only for comparison, uses the test set)
def highlight_style(kind):
    """Matplotlib marker settings for the 'cv' (filled) or 'test' (hollow) diamond."""
    if kind == "cv":
        return {"marker": "D", "markersize": 11, "markerfacecolor": INK,
                "markeredgecolor": BG, "markeredgewidth": 1.0}
    return {"marker": "D", "markersize": 11, "markerfacecolor": "none",
            "markeredgecolor": INK, "markeredgewidth": 1.8}


# draw one highlight marker on the point of a curve at a given alpha
def mark_alpha(ax, alphas, values, alpha, kind):
    """Highlight the point of the curve at this alpha."""
    i = alphas.index(alpha)
    ax.plot(alphas[i], values[i], linestyle="none", zorder=5, **highlight_style(kind))


# one series: "test" or "train" (the line style comes from LINE_STYLES)
# dots are ringed in the background color --> they stay readable when they overlap a line
def draw_curve(ax, alphas, values, series):
    """Draw one series as a 2px line, colored and styled according to the settings."""
    color = TEST_COLOR if series == "test" else TRAIN_COLOR
    ax.plot(alphas, values, color=color, linewidth=2, markeredgecolor=BG,
            markeredgewidth=1.2, zorder=3, **LINE_STYLES[LINES][series])


# legend entry that looks exactly like one of the series
def series_handle(series, label):
    """Legend entry for the 'test' or 'train' series."""
    color = TEST_COLOR if series == "test" else TRAIN_COLOR
    return Line2D([], [], color=color, linewidth=2, markeredgecolor=BG, label=label,
                  **LINE_STYLES[LINES][series])


# common look of every panel: log x-axis, light grid, title + small subtitle
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


# subtitle shown under each panel title: the two alphas that are highlighted
def alpha_subtitle(cv_alpha, best_test_alpha):
    """Text with the alpha chosen by CV and the alpha with the lowest test MSE."""
    return (rf"CV: $\alpha \approx {cv_alpha:.3g}$   |   "
            rf"lowest test MSE: $\alpha \approx {best_test_alpha:.3g}$")


# marker entries (filled / hollow diamond) shared by both legends
def marker_handles():
    """Legend entries that explain the two highlight markers."""
    return [
        Line2D([], [], linestyle="none", label=r"$\alpha$ chosen by CV (training set only)",
               **highlight_style("cv")),
        Line2D([], [], linestyle="none", label="lowest test MSE (uses the test set)",
               **highlight_style("test")),
    ]


# one legend for the whole figure, below the panels (both panels share it)
def figure_legend(fig, handles, ncol):
    """Place a shared legend at the bottom of the figure."""
    fig.legend(handles=handles, loc="lower center", ncol=ncol, frameon=False,
               fontsize=9, labelcolor=INK, columnspacing=2, handletextpad=0.6)


# title + one-line explanation at the top-left of the figure
def figure_title(fig, title, subtitle):
    """Write the figure title and a gray description under it."""
    fig.suptitle(title, x=0.015, y=0.99, ha="left", fontsize=15, fontweight="bold")
    fig.text(0.015, 0.905, subtitle, ha="left", fontsize=10, color=MUTED)
    
    
# FIGURE 1: STABILITY VS ALPHA

def plot_stability():
    """Mean |change in prediction| after removing one training point."""
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.8))  # one panel per dataset (different scales)
    for ax, (name, title) in zip(axes, DATASETS):
        data = load_sweep(name)
        alphas, values, ls_value = split_least_squares(data, "pred_change")
        _, test, _ = split_least_squares(data, "test_mse")
        cv_alpha = load_cv_alpha(name)
        best_test_alpha = test_min_alpha(alphas, test)

        ax.fill_between(alphas, values, color=TEST_COLOR, alpha=0.08)  # light wash under the curve
        draw_curve(ax, alphas, values, "test")
        # filled diamond = alpha chosen by CV, hollow diamond = alpha with the lowest test error
        mark_alpha(ax, alphas, values, cv_alpha, "cv")
        mark_alpha(ax, alphas, values, best_test_alpha, "test")

        # dashed line = least squares (alpha = 0), the unregularized baseline
        ax.axhline(ls_value, color=MUTED, linestyle="--", linewidth=1)
        ax.annotate(r"least squares ($\alpha=0$)", xy=(alphas[-1], ls_value),
                    xytext=(0, 5), textcoords="offset points", ha="right",
                    fontsize=8, color=MUTED)  # label at the right end, where the curve is low
        # headline number: how much more stable at the strongest regularization
        drop = 100 * (1 - values[-1] / ls_value)
        ax.annotate(f"-{drop:.0f}% vs least squares", xy=(alphas[-1], values[-1]),
                    xytext=(-12, -3), textcoords="offset points", ha="right", va="center",
                    fontsize=9, color=INK, fontweight="bold")  # left of the last point

        style_axis(ax, title, alpha_subtitle(cv_alpha, best_test_alpha),
                   "mean |change in prediction|")
        ax.set_ylim(bottom=0)  # start from 0 so the size of the drop is not exaggerated
    figure_title(fig, "Instability decreases as regularization increases",
                 "Average change in the test predictions when one training point "
                 "is removed (leave-one-out)")
    curve_handle = series_handle("test", "stability")
    ls_handle = Line2D([], [], color=MUTED, linestyle="--", linewidth=1,
                       label=r"least squares ($\alpha=0$)")
    figure_legend(fig, [curve_handle, ls_handle] + marker_handles(), ncol=4)
    fig.tight_layout(rect=(0, 0.07, 1, 0.95))  # leave room for title above and legend below
    fig.savefig(RESULTS_DIR / "stability_vs_alpha.png", dpi=200)
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
    plot_stability()
    plot_test_error()
    print(f"Figures saved in {RESULTS_DIR}")