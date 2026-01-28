# Direct 2×3 panel generator: draws all six subplots directly into one figure (no intermediates),
# prints correlation summaries, and saves as a single PDF vector graphic.
#
# Layout (2 rows × 3 columns; each subfigure is square):
# [ step=2500 scatter ] [ step=5000 scatter ] [ step=7500 scatter ]
# [ step=10000 scatter ] [ correlation vs step ] [ boxplot w/ mean lines ]
#
# Colors: LAI (blue), Ghost (orange). Fit lines are dashed.

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from matplotlib import font_manager

# Enlarge all fonts for readability in the generated figure
def _configure_font_family():
    """Ensure Times New Roman is available; fall back to closest serif if needed."""
    preferred = "Times New Roman"

    # Allow a local fonts/ folder next to this script to provide the font file.
    local_font_dir = Path(__file__).with_name("fonts")
    if local_font_dir.exists():
        for font_path in local_font_dir.glob("*.ttf"):
            font_manager.fontManager.addfont(str(font_path))

    available_names = {f.name for f in font_manager.fontManager.ttflist}
    if preferred in available_names:
        return preferred

    fallback_candidates = [
        "Nimbus Roman",
        "Times",
        "Liberation Serif",
        "DejaVu Serif",
    ]
    for candidate in fallback_candidates:
        if candidate in available_names:
            remaining = [name for name in fallback_candidates if name != candidate]
            plt.rcParams["font.serif"] = [candidate] + remaining
            print(
                f"Warning: Times New Roman font not found. Falling back to '{candidate}'."
            )
            return candidate

    print("Warning: Times New Roman font not found. Falling back to default font settings.")
    return plt.rcParams.get("font.family", "serif")


FONT_FAMILY = _configure_font_family()

plt.rcParams.update({
    "font.size": 18,
    "axes.titlesize": 20,
    "axes.labelsize": 18,
    "xtick.labelsize": 16,
    "ytick.labelsize": 16,
    "legend.fontsize": 12,
    "font.family": FONT_FAMILY,
})

# ---------- Config ----------
DATA_PATH = Path(__file__).parent.parent / "first_batch_scores.csv"   # 自动定位到上级目录的CSV
PANEL_PDF = Path(__file__).parent / "figure_sv.pdf"                   # 输出到当前目录
STEPS = [2500, 5000, 7500, 10000]                          # 要画散点+拟合的steps
COLOR_GHOST = "blue"
COLOR_JC = "orange"

# ---------- Load ----------
df = pd.read_csv(DATA_PATH)

def safe_corr(x, y):
    x = np.asarray(x); y = np.asarray(y)
    m = np.isfinite(x) & np.isfinite(y)
    x, y = x[m], y[m]
    if x.size < 2:
        return np.nan
    return np.corrcoef(x, y)[0, 1]

# Per-step correlation table
def _per_step_stats(group):
    return pd.Series({
        "corr_shapley_ghost": safe_corr(group["shapley"], group["ghost_score"]),
        "corr_shapley_jiacheng": safe_corr(group["shapley"], group["jiacheng_score"]),
        "n_points": len(group),
    })


try:
    grouped = (
        df.groupby("global_step")
        .apply(_per_step_stats, include_groups=False)  # pandas >= 2.1
        .reset_index()
    )
except TypeError:
    grouped = df.groupby("global_step").apply(_per_step_stats).reset_index()

# Overall summaries
avg_g = grouped["corr_shapley_ghost"].mean()
std_g = grouped["corr_shapley_ghost"].std()
avg_j = grouped["corr_shapley_jiacheng"].mean()
std_j = grouped["corr_shapley_jiacheng"].std()

print("=== Overall correlation summary (per-step) ===")
print(f"LAI vs Shapley : mean={avg_g:.6f}, std={std_g:.6f}, steps={grouped.shape[0]}")
print(f"Ghost vs Shapley : mean={avg_j:.6f}, std={std_j:.6f}, steps={grouped.shape[0]}")
print("\n=== First 10 rows of per-step table ===")
print(grouped.head(10).to_string(index=False))

def draw_scatter_fit(ax, step):
    sub = df[df["global_step"] == step]
    x = sub["shapley"].values
    y_g = sub["ghost_score"].values
    y_j = sub["jiacheng_score"].values

    scatter_lai = ax.scatter(x, y_g, s=60, alpha=0.8, color=COLOR_GHOST, marker="x", label="LAI")
    scatter_ghost = ax.scatter(x, y_j, s=60, alpha=0.8, color=COLOR_JC, marker="x", label="Ghost")

    if x.size >= 2:
        xs = np.linspace(np.nanmin(x), np.nanmax(x), 200)
        m_g, b_g = np.polyfit(x, y_g, 1)
        m_j, b_j = np.polyfit(x, y_j, 1)
        ax.plot(xs, m_g*xs + b_g, linestyle="--", linewidth=2, alpha=0.9, color=COLOR_GHOST)
        ax.plot(xs, m_j*xs + b_j, linestyle="--", linewidth=2, alpha=0.9, color=COLOR_JC)

    ax.set_xlabel("Shapley Value")
    ax.set_ylabel("Score")
    ax.grid(True, alpha=0.3)
    ax.legend(handles=[scatter_lai, scatter_ghost], loc="best", title=f"Step = {step}", title_fontsize=14)
    try:
        ax.set_box_aspect(1)  # 正方形子图（matplotlib >= 3.4）
    except Exception:
        pass

# ---------- Create panel directly ----------
fig, axes = plt.subplots(2, 3, figsize=(15, 10))
axs = axes.flatten()

labels = list("ABCDEF")
for idx, (label, ax) in enumerate(zip(labels, axs)):
    if idx < 4:
        x_pos, y_pos = 0.05, 0.95
        ha, va = "left", "top"
    else:
        x_pos, y_pos = 0.95, 0.05
        ha, va = "right", "bottom"

    ax.text(
        x_pos,
        y_pos,
        label,
        transform=ax.transAxes,
        fontsize=20,
        fontfamily=FONT_FAMILY,
        fontweight="bold",
        va=va,
        ha=ha,
    )

# 1-4: four steps scatter + dashed fits
for i, step in enumerate(STEPS):
    draw_scatter_fit(axs[i], step)

# 5: correlation vs step
ax5 = axs[4]
ax5.plot(grouped["global_step"], grouped["corr_shapley_ghost"], label="Shapley vs LAI Score", color=COLOR_GHOST)
ax5.plot(grouped["global_step"], grouped["corr_shapley_jiacheng"], label="Shapley vs Ghost Score", color=COLOR_JC)
ax5.set_xlabel("Step")
ax5.set_ylabel("Correlation Coefficient")
ax5.grid(True, alpha=0.3)
ax5.legend(loc="best")
try:
    ax5.set_box_aspect(1)
except Exception:
    pass

# 6: violin plot with colored bodies + dashed mean lines
ax6 = axs[5]
data_violin = [
    grouped["corr_shapley_ghost"].dropna(),
    grouped["corr_shapley_jiacheng"].dropna(),
]
violin = ax6.violinplot(
    data_violin,
    positions=[1, 2],
    widths=0.8,
    showmeans=False,
    showmedians=False,
    showextrema=False,
)

for body, color in zip(violin["bodies"], [COLOR_GHOST, COLOR_JC]):
    body.set_facecolor(color)
    body.set_alpha(0.5)
    body.set_edgecolor(color)
    body.set_linewidth(1)

# Means (dashed)
line_lai = ax6.axhline(avg_g, color=COLOR_GHOST, linestyle="--", linewidth=1.6, label="LAI Mean Correlation")
line_ghost = ax6.axhline(avg_j, color=COLOR_JC, linestyle="--", linewidth=1.6, label="Ghost Mean Correlation")
ax6.set_xlim(0.5, 2.5)
ax6.set_xticks([1, 2])
ax6.set_xticklabels(["LAI", "Ghost"])

ax6.set_ylabel("Correlation Coefficient")
ax6.grid(True, axis="y", alpha=0.3)
ax6.legend(handles=[line_lai, line_ghost], loc="best")
try:
    ax6.set_box_aspect(1)
except Exception:
    pass

plt.tight_layout()
for ax in axs:
    try:
        ax.set_box_aspect(1)
    except Exception:
        pass
fig.savefig(PANEL_PDF, format="pdf")  # 矢量PDF
plt.close(fig)

print(f"\nSaved panel PDF to: {PANEL_PDF}")
