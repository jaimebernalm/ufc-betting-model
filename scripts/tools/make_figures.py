"""Generate README/docs figures from the regenerated metrics artifacts."""

import json

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

from ufc_pred.paths import METRICS, ROOT

OUT = ROOT / "docs" / "figures"
OUT.mkdir(parents=True, exist_ok=True)

# Palette (validated: see dataviz skill reference palette, light surface)
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"
BLUE = "#2a78d6"
ORANGE = "#eb6834"
RED = "#e34948"

mpl.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": ["Helvetica Neue", "Helvetica", "Arial", "DejaVu Sans"],
        "figure.facecolor": SURFACE,
        "axes.facecolor": SURFACE,
        "savefig.facecolor": SURFACE,
        "text.color": INK,
        "axes.labelcolor": INK2,
        "xtick.color": MUTED,
        "ytick.color": MUTED,
        "axes.edgecolor": AXIS,
        "grid.color": GRID,
        "axes.grid": True,
        "grid.linewidth": 0.8,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.linewidth": 1.0,
        "font.size": 10,
        "figure.dpi": 200,
    }
)


def _style(ax, ylabel=None, title=None):
    ax.set_axisbelow(True)
    if ylabel:
        ax.set_ylabel(ylabel, fontsize=9.5, color=INK2)
    if title:
        ax.set_title(title, fontsize=11, color=INK, weight="bold", pad=10, loc="left")


# ── 1. Early stopping: log-loss vs ROI, small multiples (never dual-axis) ──────
fig, axes = plt.subplots(1, 2, figsize=(7.6, 3.3))
labels = ["early stop\nON", "early stop\nOFF"]
x = np.arange(2)

ll = [0.6189, 0.6452]
ax = axes[0]
ax.bar(x, ll, width=0.5, color=[BLUE, ORANGE], zorder=3)
for xi, v in zip(x, ll, strict=True):
    ax.text(xi, v + 0.002, f"{v:.4f}", ha="center", fontsize=9.5, color=INK, weight="bold")
ax.set_xticks(x, labels, fontsize=9.5, color=INK2)
ax.set_ylim(0.60, 0.665)
_style(ax, "validation log-loss  (lower better)", "Predictive quality gets worse")
ax.grid(axis="x", visible=False)

roi = [-8.26, 2.86]
ax = axes[1]
ax.bar(x, roi, width=0.5, color=[BLUE, ORANGE], zorder=3)
ax.axhline(0, color=AXIS, lw=1.2, zorder=4)
for xi, v in zip(x, roi, strict=True):
    off = 0.9 if v > 0 else -2.2
    ax.text(xi, v + off, f"{v:+.2f}%", ha="center", fontsize=9.5, color=INK, weight="bold")
ax.set_xticks(x, labels, fontsize=9.5, color=INK2)
ax.set_ylim(-12, 7)
_style(ax, "validation ROI, Kalshi fees", "…while betting return flips positive")
ax.grid(axis="x", visible=False)

fig.suptitle(
    "Same model, same seed — only early stopping differs",
    fontsize=12,
    weight="bold",
    color=INK,
    x=0.012,
    ha="left",
    y=1.0,
)
fig.tight_layout(rect=[0, 0, 1, 0.93])
fig.savefig(OUT / "early_stopping_tradeoff.png", bbox_inches="tight")
plt.close(fig)

# ── 2. Post-cutoff ROI by venue/window, with CIs ──────────────────────────────
rows = [
    ("Polymarket\nno debut", 156, 22.5, 4.2, 41.5),
    ("Kalshi\nthrough 2026-05-16", 42, 39.0, -9.4, 103.4),
    ("Kalshi\nafter 2026-05-16", 60, -30.3, -52.8, -5.2),
]
fig, ax = plt.subplots(figsize=(7.0, 3.6))
y = np.arange(len(rows))[::-1]
for yi, (_lab, _n, roi_v, lo, hi) in zip(y, rows, strict=True):
    c = BLUE if roi_v > 0 else RED
    ax.plot([lo, hi], [yi, yi], color=c, lw=2, alpha=0.45, solid_capstyle="round", zorder=3)
    ax.plot([lo, lo], [yi - 0.1, yi + 0.1], color=c, lw=2, zorder=3)
    ax.plot([hi, hi], [yi - 0.1, yi + 0.1], color=c, lw=2, zorder=3)
    ax.scatter([roi_v], [yi], s=95, color=c, zorder=5, edgecolor=SURFACE, linewidth=2)
    ax.text(
        roi_v,
        yi + 0.26,
        f"{roi_v:+.1f}%",
        ha="center",
        fontsize=10,
        color=INK,
        weight="bold",
        zorder=6,
    )
ax.axvline(0, color=AXIS, lw=1.2, zorder=2)
ax.set_yticks(y, [f"{r[0]}   (n={r[1]})" for r in rows], fontsize=9.5, color=INK2)
ax.set_xlim(-70, 115)
ax.set_ylim(-0.6, len(rows) - 1 + 0.85)
ax.set_xlabel("ROI %  ·  bars are 95% confidence intervals", fontsize=9.5, color=INK2)
ax.grid(axis="y", visible=False)
_style(ax, None, "The edge decayed: frozen model, strictly out-of-sample")
ax.title.set_position((0, 1.06))
fig.tight_layout()
fig.savefig(OUT / "edge_decay.png", bbox_inches="tight")
plt.close(fig)

# ── 3. Seed variance ──────────────────────────────────────────────────────────
sv = json.load(open(METRICS / "seed_variance_test.json"))
finals = sorted(s["final"] for s in sv["2024-01-01"]["seeds"])
fig, ax = plt.subplots(figsize=(7.0, 2.5))
ax.scatter(
    finals,
    np.zeros(len(finals)),
    s=110,
    color=BLUE,
    alpha=0.75,
    zorder=4,
    edgecolor=SURFACE,
    linewidth=1.6,
)
ax.set_xscale("log")
ax.set_yticks([])
ax.set_ylim(-1, 1.1)
lo, hi = finals[0], finals[-1]
ax.annotate(
    "",
    xy=(lo, 0.5),
    xytext=(hi, 0.5),
    arrowprops={"arrowstyle": "<->", "color": MUTED, "lw": 1.2},
)
ax.text(
    np.sqrt(lo * hi),
    0.62,
    f"{hi / lo:.0f}× spread",
    ha="center",
    fontsize=10.5,
    color=INK,
    weight="bold",
)
for v, ha in ((lo, "right"), (hi, "left")):
    ax.text(
        v, -0.55, f"${v / 1e6:.2f}M" if v >= 1e6 else f"${v / 1000:,.0f}k", ha=ha, fontsize=9.5, color=INK2
    )
ax.set_xlabel(
    "final bankroll, log scale  ·  10 seeds, identical data & hyperparameters", fontsize=9.5, color=INK2
)
ax.grid(axis="y", visible=False)
_style(ax, None, "One seed is a sample, not a model")
fig.tight_layout()
fig.savefig(OUT / "seed_variance.png", bbox_inches="tight")
plt.close(fig)

for p in sorted(OUT.glob("*.png")):
    print(f"{p.relative_to(ROOT)}  {p.stat().st_size / 1024:.0f} KB")
