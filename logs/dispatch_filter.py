"""Recreate the reference plot's styling using the paragraph's constraints.

Install: python -m pip install matplotlib
Run:     python reproduce_dispatch_graph.py
View:    python reproduce_dispatch_graph.py --show

"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import FixedLocator, PercentFormatter


# Edit these illustrative values when exact condition-specific data is available.
#                         dispatch rate, accuracy, color
CONDITIONS = {
    "No Context":       (1.00, 0.74, "#E69F00"),
    "With Context":     (1.00, 0.79, "#009E73"),
    "Filter":           (0.50, 0.91, "#0072B2"),
    "Filter + Context": (0.50, 0.96, "#CC79A7"),
}

# Standard deviation of the faint markers around each condition's mean, in
# fractions (0.10 = 10 percentage points). It is applied to accuracy and, for
# conditions with a dispatch rate below 100%, to dispatch rate as well.
STD_DEV = {
    "No Context":       0.12,
    "With Context":     0.13,
    "Filter":           0.10,
    "Filter + Context": 0.12,
}
N_FAINT_POINTS = 10   # faint markers drawn per condition
FAINT_ALPHA = 0.2
SEED = 0              # fixed so the figure is reproducible


def faint_points(rng, rate, accuracy, std):
    """Sample points around the mean; both axes are clipped to [0, 1]."""
    ys = rng.normal(accuracy, std, N_FAINT_POINTS)
    # Dispatch rate cannot exceed 100%, so conditions at 100% stay on that line.
    xs = rng.normal(rate, std, N_FAINT_POINTS) if rate < 1.0 else [rate] * N_FAINT_POINTS
    return np.clip(xs, 0, 1), np.clip(ys, 0, 1)


def make_figure():
    """Return the styled figure and axes; values are fractions, not percents."""
    with plt.rc_context({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Times", "STIXGeneral", "DejaVu Serif"],
        "font.size": 16,
        "axes.labelsize": 16,
        "axes.edgecolor": "#333333",
        "axes.linewidth": 1.5,
        "xtick.major.size": 6,
        "ytick.major.size": 6,
        "xtick.major.width": 1.3,
        "ytick.major.width": 1.3,
    }):
        fig, ax = plt.subplots(figsize=(9.47, 6.10), dpi=100)
        fig.subplots_adjust(left=0.139, right=0.978, bottom=0.12, top=0.935)
        fig.patch.set_facecolor("white")
        ax.set_facecolor("white")

        # The reference uses guides only at 100%, rather than a full grid.
        ax.axhline(1.0, color="#dddddd", linestyle="--", linewidth=1.0, zorder=0)
        ax.axvline(1.0, color="#dddddd", linestyle="--", linewidth=1.0, zorder=0)

        rng = np.random.default_rng(SEED)
        for label, (rate, accuracy, color) in CONDITIONS.items():
            xs, ys = faint_points(rng, rate, accuracy, STD_DEV[label])
            ax.scatter(xs, ys, s=150, color=color, alpha=FAINT_ALPHA,
                       edgecolors="none", zorder=2)
            ax.scatter(rate, accuracy, s=275, color=color,
                       edgecolors="#444444", linewidths=1.6,
                       label=label, zorder=3)

        ax.set_xlim(-0.05, 1.15)
        ax.set_ylim(0, 1.15)
        ticks = [0, 0.2, 0.4, 0.6, 0.8, 1.0]
        for axis in (ax.xaxis, ax.yaxis):
            axis.set_major_locator(FixedLocator(ticks))
            axis.set_major_formatter(PercentFormatter(xmax=1, decimals=0))
        ax.tick_params(axis="both", pad=6)
        ax.set_xlabel("Dispatch Rate (calls sent to robot / total)")
        ax.set_ylabel("Dispatch Accuracy")
        ax.spines[["top", "right"]].set_visible(False)

        legend = ax.legend(title="Condition", loc="lower right",
                           fontsize=14, title_fontsize=14,
                           frameon=True, fancybox=True, framealpha=1,
                           borderpad=0.3, labelspacing=0.25,
                           handletextpad=0.6, borderaxespad=0.4)
        legend.get_frame().set_edgecolor("#d0d0d0")
        legend.get_frame().set_linewidth(1.5)
        return fig, ax


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("dispatch_graph.pdf"))
    parser.add_argument("--show", action="store_true")
    args = parser.parse_args()
    fig, _ = make_figure()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=300, facecolor="white")
    print(f"Saved {args.output.resolve()}")
    print("Illustrative values: see the assumptions at the top of this script.")
    if args.show:
        plt.show()
    plt.close(fig)


if __name__ == "__main__":
    main()
