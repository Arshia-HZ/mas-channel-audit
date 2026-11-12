"""Exploratory figures (PREREG 10.2, item 9): accuracy against tokens, change types
and the source of change.

The numbers behind each figure come from small pure functions, so they are testable
without drawing; matplotlib is imported only when a figure is drawn (install it with
``pip install -e .[figures]``). Every figure carries the exploratory footer. Figure 1
is the arm table in ``analysis.md``; the teams figure (5) belongs to M8.
"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path

from channel_audit.analysis.decompose import CHANGE_TYPES

FOOTER = "EXPLORATORY - descriptive, not a preregistered result (PREREG 10.2, item 9)"
FIGURE_NAMES = (
    "fig2_accuracy_vs_tokens.png",
    "fig3_change_types.png",
    "fig4_source_of_change.png",
)
CHANGES_SHOWN = ("wrong_to_correct", "correct_to_wrong", "wrong_to_other_wrong")
PART_COLORS = {"spontaneous": "#8c8c8c", "form": "#d98c2b", "content": "#3b7dc4"}


# ---- the numbers behind the figures ------------------------------------------


def accuracy_vs_tokens_data(analysis: Mapping) -> dict:
    """The self-consistency curve and the arms as (tokens, accuracy) points.

    One self-consistency sample costs an arm's tokens divided by its matched sample
    count; that is the same for every arm, so the curve's x axis is k times it.
    """
    arms = analysis["analyses"]["main"]["arms"]
    costs = [a["tokens"] / a["matched_k"] for a in arms.values()]
    per_sample = sum(costs) / len(costs)
    curve = sorted(
        (int(k) * per_sample, acc) for k, acc in analysis["sc_curve"].items()
    )
    return {
        "per_sample_tokens": per_sample,
        "curve": curve,
        "arms": {name: (a["tokens"], a["accuracy"]) for name, a in arms.items()},
    }


def change_type_data(decomposition: Mapping) -> dict:
    return {
        arm: {t: row["shares"][t]["mean"] for t in CHANGE_TYPES}
        for arm, row in decomposition["changes"].items()
    }


def source_data(decomposition: Mapping, metric: str = "net") -> dict:
    return {
        arm: {k: parts[k]["mean"] for k in ("spontaneous", "form", "content")}
        for arm, parts in decomposition["source"][metric].items()
    }


# ---- drawing -----------------------------------------------------------------


def _pyplot():
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise ImportError(
            "drawing needs matplotlib: pip install -e .[figures]"
        ) from exc
    return plt


def _finish(fig, plt, path: str | Path) -> None:
    fig.text(0.01, 0.005, FOOTER, fontsize=7, color="#a33", ha="left", va="bottom")
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    fig.savefig(path, dpi=150)
    plt.close(fig)


def zoom_box(data: Mapping) -> tuple[tuple[float, float], tuple[float, float]]:
    """Axis limits around the arms that score close to the best one.

    The arms cluster within a few points of each other, so a full-range plot cannot
    separate them; the zoom panel shows them with the self-consistency points near.
    """
    points = list(data["arms"].values())
    best = max(y for _, y in points)
    near = [(x, y) for x, y in points if y >= best - 0.03]
    x0, x1 = min(x for x, _ in near), max(x for x, _ in near)
    pad = 0.15 * ((x1 - x0) or 1.0)
    inside = [y for x, y in data["curve"] if x0 - 3 * pad <= x <= x1 + 3 * pad]
    ys = [y for _, y in near] + inside
    return (x0 - 3 * pad, x1 + 3 * pad), (min(ys) - 0.008, max(ys) + 0.008)


def plot_accuracy_vs_tokens(analysis: Mapping, path: str | Path) -> None:
    plt = _pyplot()
    from matplotlib.patches import Rectangle

    d = accuracy_vs_tokens_data(analysis)
    (x0, x1), (y0, y1) = zoom_box(d)
    fig, (full, zoom) = plt.subplots(1, 2, figsize=(13, 5.2))
    markers = "os^Dv<>p*hX"
    for ax in (full, zoom):
        xs, ys = zip(*d["curve"], strict=True)
        ax.plot(xs, ys, "-o", color="#555", label="self-consistency (k samples)")
        for k, (x, y) in enumerate(d["curve"], start=1):
            ax.annotate(
                f"k={k}",
                (x, y),
                textcoords="offset points",
                xytext=(4, -11),
                fontsize=7,
            )
        for i, (name, (x, y)) in enumerate(d["arms"].items()):
            ax.scatter(
                [x],
                [y],
                s=55,
                marker=markers[i % len(markers)],
                color=f"C{i}",
                zorder=3,
                label=name,
            )
        ax.set_xlabel("mean generated tokens per replicate")
        ax.set_ylabel("accuracy (plurality credit)")
    full.add_patch(
        Rectangle((x0, y0), x1 - x0, y1 - y0, fill=False, ls="--", color="#777")
    )
    full.set_title(f"{analysis['model_id']}: all arms")
    zoom.set_xlim(x0, x1)
    zoom.set_ylim(y0, y1)
    zoom.set_title("zoom on the arms near the best score (dashed box on the left)")
    zoom.legend(loc="upper left", fontsize=8)
    _finish(fig, plt, path)


def plot_change_types(decomposition: Mapping, path: str | Path) -> None:
    plt = _pyplot()
    arms = list(decomposition["changes"])
    fig, ax = plt.subplots(figsize=(9, 5))
    width = 0.8 / len(CHANGES_SHOWN)
    colors = {
        "wrong_to_correct": "#3b9a5a",
        "correct_to_wrong": "#c0392b",
        "wrong_to_other_wrong": "#8c8c8c",
    }
    for j, t in enumerate(CHANGES_SHOWN):
        means = [decomposition["changes"][a]["shares"][t]["mean"] for a in arms]
        lo = [
            m - decomposition["changes"][a]["shares"][t]["ci95_low"]
            for m, a in zip(means, arms, strict=True)
        ]
        hi = [
            decomposition["changes"][a]["shares"][t]["ci95_high"] - m
            for m, a in zip(means, arms, strict=True)
        ]
        xs = [i + (j - 1) * width for i in range(len(arms))]
        ax.bar(
            xs,
            means,
            width,
            yerr=[lo, hi],
            capsize=2,
            label=t.replace("_", " "),
            color=colors[t],
        )
    ax.set_xticks(range(len(arms)), arms, rotation=30, ha="right")
    ax.set_ylabel("share of agents")
    ax.set_title("What each channel changes (95% intervals, unadjusted)")
    ax.legend()
    _finish(fig, plt, path)


def plot_source_of_change(
    decomposition: Mapping, path: str | Path, metric: str = "net"
) -> None:
    plt = _pyplot()
    data = source_data(decomposition, metric)
    totals = {a: decomposition["source"][metric][a]["total"]["mean"] for a in data}
    arms = list(data)
    parts = ("spontaneous", "form", "content")
    fig, ax = plt.subplots(figsize=(9, 5))
    width = 0.8 / len(parts)
    for j, part in enumerate(parts):
        xs = [i + (j - 1) * width for i in range(len(arms))]
        ax.bar(
            xs,
            [data[a][part] for a in arms],
            width,
            label=part,
            color=PART_COLORS[part],
        )
    ax.scatter(
        range(len(arms)),
        [totals[a] for a in arms],
        marker="D",
        color="black",
        zorder=3,
        label="total",
    )
    ax.axhline(0, color="#333", linewidth=0.6)
    ax.set_xticks(range(len(arms)), arms, rotation=30, ha="right")
    ax.set_ylabel(f"{metric.replace('_', ' ')} (share of agents)")
    ax.set_title(
        "Source of change: spontaneous + form (placebo) + content (arm minus placebo)"
    )
    ymin, ymax = ax.get_ylim()
    ax.set_ylim(ymin, ymax + 0.25 * (ymax - ymin))  # room for the legend
    ax.legend(ncol=4, loc="upper left")
    _finish(fig, plt, path)


def make_all(
    analysis: Mapping, decomposition: Mapping, out_dir: str | Path
) -> list[Path]:
    """Write the three figures; refuse to overwrite any existing file."""
    out = Path(out_dir)
    paths = [out / name for name in FIGURE_NAMES]
    for p in paths:
        if p.exists():
            raise FileExistsError(f"{p} already exists: figures are never overwritten")
    out.mkdir(parents=True, exist_ok=True)
    plot_accuracy_vs_tokens(analysis, paths[0])
    plot_change_types(decomposition, paths[1])
    plot_source_of_change(decomposition, paths[2])
    return paths
