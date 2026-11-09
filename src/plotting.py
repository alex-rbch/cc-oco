import json
import math
import os
from pathlib import Path
import statistics
import tempfile

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "cc-oco-matplotlib"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator, ScalarFormatter
from scipy.stats import t

STYLES = {
    "uniform": ("Uniform-weight (ours)", "#0072B2", "o", "-"),
    "pareto": ("Pareto (prior work)", "#D55E00", "s", "--"),
    "full": ("Full tracking", "#626873", None, ":"),
}


def statistics_for(values):
    mean = statistics.fmean(values)
    count = len(values)
    std = statistics.stdev(values) if count > 1 else None
    half_width = float(t.ppf(0.975, count - 1)) * std / math.sqrt(count) if count > 1 else None
    return dict(n=count, mean=mean, std=std,
                ci95_low=mean - half_width if half_width is not None else None,
                ci95_high=mean + half_width if half_width is not None else None)


def summarize(rows, config):
    from .experiment import CAPACITIES, FAMILIES, PROBLEMS

    groups = []
    for family in FAMILIES:
        for problem in PROBLEMS:
            for scheduler in STYLES:
                for capacity in CAPACITIES:
                    selected = [row for row in rows if row["loss_family"] == family
                                and row["problem_type"] == problem and row["scheduler"] == scheduler
                                and int(row["capacity"]) == capacity]
                    if {int(row["seed"]) for row in selected} != set(config.final_seeds) or len(selected) != len(config.final_seeds):
                        raise ValueError("Plotting requires one completed run per seed and condition")
                    groups.append(dict(loss_family=family, problem=problem, scheduler=scheduler, capacity=capacity,
                                       **statistics_for([float(row["normalized_regret"]) for row in selected])))
    for family in FAMILIES:
        for problem in PROBLEMS:
            full = [group["mean"] for group in groups if group["loss_family"] == family
                    and group["problem"] == problem and group["scheduler"] == "full"]
            if len(set(full)) != 1:
                raise ValueError("Full-tracking results differ across capacities")
    return dict(horizon=config.horizon, repetitions=len(config.final_seeds), capacities=list(CAPACITIES),
                uncertainty="Pointwise 95% Student-t confidence intervals for the mean across paired repetitions (n-1 degrees of freedom); omitted for one repetition.",
                groups=groups)


def plot_results(rows, config, output):
    from .experiment import FAMILIES, PROBLEMS, atomic_text

    summary = summarize(rows, config)
    plt.rcParams.update({
        "font.family": "serif", "font.serif": ["STIXGeneral"], "mathtext.fontset": "stix",
        "font.size": 9, "axes.labelsize": 8.5, "legend.fontsize": 8.5,
        "xtick.labelsize": 8, "ytick.labelsize": 8,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.edgecolor": "#7A828B", "axes.linewidth": 0.6,
        "pdf.fonttype": 42, "ps.fonttype": 42, "lines.linewidth": 1.25,
    })
    fig, axes = plt.subplots(2, 2, figsize=(5.5, 3.2), sharex=True)
    for row, family in enumerate(FAMILIES):
        for col, problem in enumerate(PROBLEMS):
            ax = axes[row, col]
            for scheduler, (label, color, marker, linestyle) in STYLES.items():
                groups = [group for group in summary["groups"] if group["loss_family"] == family
                          and group["problem"] == problem and group["scheduler"] == scheduler]
                xs = [group["capacity"] for group in groups]
                means = [group["mean"] for group in groups]
                if scheduler == "full":
                    ax.axhline(means[0], color=color, linestyle=linestyle, linewidth=1.1, zorder=2)
                else:
                    errors = [[group["mean"] - group["ci95_low"] for group in groups],
                              [group["ci95_high"] - group["mean"] for group in groups]] if len(config.final_seeds) > 1 else None
                    ax.errorbar(xs, means, yerr=errors, color=color, marker=marker, linestyle=linestyle,
                                markersize=3.8, markerfacecolor="white", markeredgewidth=0.9,
                                capsize=1.9, elinewidth=0.65, zorder=4 if scheduler == "uniform" else 3)
            ax.set_xticks(xs, labels=[str(x) for x in xs])
            ax.margins(x=0.09, y=0.15)
            lower_values = [group["ci95_low"] if group["ci95_low"] is not None else group["mean"]
                            for group in summary["groups"] if group["loss_family"] == family and group["problem"] == problem]
            if ax.get_ylim()[0] < 0 and min(lower_values) >= 0:
                ax.set_ylim(bottom=0)
            ax.yaxis.set_major_locator(MaxNLocator(nbins=3, min_n_ticks=3, steps=[1, 2, 2.5, 5, 10]))
            formatter = ScalarFormatter(useOffset=False)
            formatter.set_scientific(False)
            ax.yaxis.set_major_formatter(formatter)
            ax.set_axisbelow(True)
            ax.grid(axis="y", color="#E4E8ED", linewidth=0.45)
            ax.tick_params(length=3, width=0.6, pad=2.5)
    for ax, title in zip(axes[0], ("First-order feedback", "Bandit feedback")):
        ax.set_title(title, fontsize=9.2, pad=4.5)
    fig.supxlabel(r"Capacity $C$", x=0.5775, y=0.03, fontsize=8.5, fontweight="bold")
    fig.supylabel(r"Mean regret $R_T/T$", x=0.01, y=0.4875, fontsize=8.5, fontweight="bold")
    fig.subplots_adjust(left=0.17, right=0.985, bottom=0.135, top=0.84, wspace=0.24, hspace=0.2)
    for ax, label in zip(axes[:, 0], ("Convex", "Strongly convex")):
        bounds = ax.get_position()
        fig.text(0.06, (bounds.y0 + bounds.y1) / 2, label, rotation=90,
                 ha="center", va="center", fontsize=9)
    handles = [Line2D([], [], color=color, linestyle=linestyle, marker=marker, markersize=3.8,
                      markerfacecolor="white", markeredgewidth=0.9,
                      linewidth=1.1 if scheduler == "full" else 1.25)
               for scheduler, (_, color, marker, linestyle) in STYLES.items()]
    fig.legend(handles, [style[0] for style in STYLES.values()], loc="upper center", ncol=3,
               bbox_to_anchor=(0.52, 0.99), frameon=False, columnspacing=1.6, handlelength=2.4, handletextpad=0.6)
    fig.savefig(output / "figure3.pdf", facecolor="white", metadata={"CreationDate": None})
    fig.savefig(output / "figure3.png", facecolor="white", dpi=220)
    plt.close(fig)
    atomic_text(output / "summary.json", json.dumps(summary, indent=2, allow_nan=False) + "\n")
