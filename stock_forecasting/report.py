"""Write backtest results to CSV and draw the summary chart."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

LABELS = {
    "naive": "Naive (no change)",
    "ridge": "Ridge regression",
    "arima": "ARIMA(5,0,2)",
    "lstm": "LSTM",
    "transformer": "Transformer",
}
BAR, MUTED, INK, INK_2, GRID, SURFACE = "#2a78d6", "#a3a29c", "#0b0b0b", "#52514e", "#e6e5e0", "#fcfcfb"


def write_reports(preds: pd.DataFrame, per_stock: pd.DataFrame, summary: pd.DataFrame, out_dir: str | Path) -> None:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    preds.to_csv(out / "predictions.csv", index=False)
    per_stock.round(5).to_csv(out / "metrics_by_stock.csv", index=False)
    summary.round(5).to_csv(out / "summary.csv", index=False)
    plot_summary(summary, out / "backtest_summary.png")


def plot_summary(summary: pd.DataFrame, path: str | Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    models = summary[summary.model != "naive"].iloc[::-1]
    labels = [LABELS.get(m, m) for m in models.model]
    plt.rcParams.update(
        {
            "font.size": 11,
            "axes.edgecolor": GRID,
            "text.color": INK,
            "axes.labelcolor": INK_2,
            "xtick.color": INK_2,
            "ytick.color": INK,
        }
    )
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 3.6), sharey=True, facecolor=SURFACE)

    skill = models.skill_vs_naive * 100
    ax1.barh(labels, skill, color=BAR, height=0.5)
    ax1.axvline(0, color=INK_2, lw=1)
    ax1.set_title("Error vs naive forecast (mean across stocks)", loc="left", fontsize=12)
    ax1.set_xlabel("% lower error than naive (right is better)")
    for y, v in enumerate(skill):
        ax1.annotate(
            f"{v:+.1f}%",
            (v, y),
            xytext=(4 if v >= 0 else -4, 0),
            textcoords="offset points",
            va="center",
            ha="left" if v >= 0 else "right",
            color=INK_2,
            fontsize=10,
        )
    lim = max(abs(skill).max() * 1.6, 1)
    ax1.set_xlim(-lim, lim)

    acc = models.directional_accuracy * 100
    ys = range(len(models))
    always_up = summary.up_day_share.iloc[0] * 100
    ax2.axvline(50, color=INK_2, lw=1)
    ax2.axvline(always_up, color=MUTED, lw=1.5, ls="--")
    ax2.hlines(ys, 50, acc, color=BAR, lw=2)
    ax2.plot(acc, ys, "o", color=BAR, markersize=9, markeredgecolor=SURFACE, markeredgewidth=2)
    ax2.annotate(
        f"always 'up': {always_up:.1f}%",
        (always_up, -0.75),
        xytext=(5, 0),
        textcoords="offset points",
        color=INK_2,
        fontsize=10,
        va="center",
    )
    ax2.annotate(
        "coin flip",
        (50, -0.75),
        xytext=(-5, 0),
        textcoords="offset points",
        color=INK_2,
        fontsize=10,
        va="center",
        ha="right",
    )
    ax2.set_title("Direction called correctly", loc="left", fontsize=12)
    ax2.set_xlabel("% of days")
    for y, v in zip(ys, acc, strict=True):
        ax2.annotate(
            f"{v:.1f}%", (v, y), xytext=(0, 9), textcoords="offset points", ha="center", color=INK_2, fontsize=10
        )
    ax2.set_ylim(-1.1, len(models) - 0.5)
    ax2.set_xlim(47, 57)

    for ax in (ax1, ax2):
        ax.set_facecolor(SURFACE)
        ax.grid(axis="x", color=GRID, lw=0.8)
        ax.set_axisbelow(True)
        for side in ("top", "right", "left"):
            ax.spines[side].set_visible(False)
        ax.tick_params(axis="y", length=0)
    fig.tight_layout()
    fig.savefig(path, dpi=150, facecolor=SURFACE)
    plt.close(fig)
