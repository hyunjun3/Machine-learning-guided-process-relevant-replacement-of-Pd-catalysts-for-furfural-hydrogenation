"""Generate manuscript Figure 4a."""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from figures.common import *  # noqa: F403

globals().update(load_ml_context())  # noqa: F405


#%% Fig. 4-(a)
data = {
    'Catalyst': [
        '1Ru–4Ni', '1Pd–4Ni', '1Ir–4Ni', '1Pt–4Ni', '1Rh–4Ni', '5Pd',
        '1Co–4Ni', '1Fe–4Ni', '1Cu–4Ni', '1Ca–4Ni',
        '1Zn–4Ni', '1Re–4Ni', '5Ni', '5Re'
    ],
    'Experimental': [
        80.45, 73.34, 73.89, 55.48, 66.48, 87.89,
        10.83, 49.53, 37.57, 31.37, 26.25, 84.26,
        32.58, 15.26
    ],
    'Predicted': [
        67.25, 73.04, 73.50, 55.19, 53.42, 78.86,
        11.08, 49.44, 37.50, 41.21, 39.57, 80.12,
        32.97, 49.25
    ],
    'noble': [
        True, True, True, True, True, True,
        False, False, False, False, False, False,
        False, False
    ]
}

df = pd.DataFrame(data)

noble_idx = df.index[df['noble']].to_numpy()
noble_free_idx = df.index[~df['noble']].to_numpy()
#####################################################################################################################
MM_TO_INCH = 1 / 25.4

# Full-width figure
FIG_WIDTH = 180 * MM_TO_INCH
FIG_HEIGHT = 115 * MM_TO_INCH

# Nature figure typography
# Panel label: 8 pt
# All other text: 5–7 pt
FS_PANEL = 12.0
FS_AXIS = 10.0
FS_GROUP = 9.5
FS_TICK = 7
FS_VALUE = 6.0
FS_LEGEND = 7.5

mpl.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': 'Arial',

    'font.size': FS_TICK,
    'axes.labelsize': FS_AXIS,
    'axes.linewidth': 0.6,

    'xtick.labelsize': FS_TICK,
    'ytick.labelsize': FS_TICK,
    'xtick.major.size': 2.5,
    'ytick.major.size': 2.5,
    'xtick.major.width': 0.6,
    'ytick.major.width': 0.6,
    'xtick.direction': 'out',
    'ytick.direction': 'out',

    'legend.fontsize': FS_LEGEND,
    'legend.frameon': False,

    # Editable embedded text in vector files
    'pdf.fonttype': 42,
    'ps.fonttype': 42,
    'svg.fonttype': 'none',

    'savefig.facecolor': 'white'
})


# =============================================================================
# 2. Accessible colours
# =============================================================================
# Colour-blind-accessible palette
COLOR_EXP_NOBLE = '#0072B2'     # dark blue
COLOR_PRED_NOBLE = '#56B4E9'    # sky blue
COLOR_EXP_FREE = '#D55E00'      # vermillion
COLOR_PRED_FREE = '#E69F00'     # orange


# =============================================================================
# 3. Figure
# =============================================================================
fig, ax = plt.subplots(
    figsize=(FIG_WIDTH, FIG_HEIGHT),
    dpi=300
)

x = np.arange(len(df))
bar_width = 0.38


# =============================================================================
# 4. Bars
# =============================================================================
bars_exp_noble = ax.bar(
    x[noble_idx] - bar_width / 2,
    df.loc[noble_idx, 'Experimental'],
    width=bar_width,
    color=COLOR_EXP_NOBLE,
    edgecolor='black',
    linewidth=0.35,
    label='Experimental (noble-metal)',
    zorder=3
)

bars_pred_noble = ax.bar(
    x[noble_idx] + bar_width / 2,
    df.loc[noble_idx, 'Predicted'],
    width=bar_width,
    color=COLOR_PRED_NOBLE,
    edgecolor='black',
    linewidth=0.35,
    label='Predicted (noble-metal)',
    zorder=3
)

bars_exp_free = ax.bar(
    x[noble_free_idx] - bar_width / 2,
    df.loc[noble_free_idx, 'Experimental'],
    width=bar_width,
    color=COLOR_EXP_FREE,
    edgecolor='black',
    linewidth=0.35,
    label='Experimental (noble-metal-free)',
    zorder=3
)

bars_pred_free = ax.bar(
    x[noble_free_idx] + bar_width / 2,
    df.loc[noble_free_idx, 'Predicted'],
    width=bar_width,
    color=COLOR_PRED_FREE,
    edgecolor='black',
    linewidth=0.35,
    label='Predicted (noble-metal-free)',
    zorder=3
)


# =============================================================================
# 5. Bar-value labels
# =============================================================================
SHOW_VALUES = True


def add_bar_labels(axis, bars):
    if not SHOW_VALUES:
        return

    for bar in bars:
        height = bar.get_height()

        if not np.isfinite(height):
            continue

        axis.text(
            bar.get_x() + bar.get_width() / 2,
            height + 1.0,
            f'{height:.1f}',
            ha='center',
            va='bottom',
            fontsize=FS_VALUE,
            fontweight='normal',
            color='black',
            rotation=0,
            clip_on=False
        )


add_bar_labels(ax, bars_exp_noble)
add_bar_labels(ax, bars_pred_noble)
add_bar_labels(ax, bars_exp_free)
add_bar_labels(ax, bars_pred_free)


# =============================================================================
# 6. Catalyst-group separation
# =============================================================================
group_boundary = len(noble_idx) - 0.5

# Background shading is omitted because it is not essential.
# Use only a simple group-separation line.
ax.axvline(
    group_boundary,
    color='0.55',
    linewidth=0.6,
    linestyle='-',
    zorder=2
)

noble_center = (
    noble_idx.min() + noble_idx.max()
) / 2

free_center = (
    noble_free_idx.min() + noble_free_idx.max()
) / 2


# Group labels: Nature recommends avoiding coloured text
ax.text(
    noble_center,
    1.015,
    'Noble-metal catalysts',
    transform=ax.get_xaxis_transform(),
    ha='center',
    va='bottom',
    fontsize=FS_GROUP,
    fontweight='bold',
    color='black',
    clip_on=False
)

ax.text(
    free_center,
    1.015,
    'Noble-metal-free catalysts',
    transform=ax.get_xaxis_transform(),
    ha='center',
    va='bottom',
    fontsize=FS_GROUP,
    fontweight='bold',
    color='black',
    clip_on=False
)


# =============================================================================
# 7. Axes
# =============================================================================
ax.set_xticks(x)

ax.set_xticklabels(
    df['Catalyst'],
    rotation=35,
    ha='right',
    rotation_mode='anchor',
    fontsize=FS_TICK
)

ax.set_xlabel(
    'Catalyst',
    fontsize=FS_AXIS,
    labelpad=5
)

ax.set_ylabel(
    'THFA yield (%)',
    fontsize=FS_AXIS,
    labelpad=4
)

ax.set_xlim(
    -0.65,
    len(df) - 0.35
)

# Dynamic upper limit with enough room for value labels
all_values = np.concatenate([
    df.loc[noble_idx, 'Experimental'].to_numpy(dtype=float),
    df.loc[noble_idx, 'Predicted'].to_numpy(dtype=float),
    df.loc[noble_free_idx, 'Experimental'].to_numpy(dtype=float),
    df.loc[noble_free_idx, 'Predicted'].to_numpy(dtype=float)
])

max_bar_value = np.nanmax(all_values)

y_upper = max(
    105,
    np.ceil((max_bar_value + 5) / 5) * 5
)

ax.set_ylim(
    0,
    y_upper
)

ax.set_yticks(
    np.arange(
        0,
        min(y_upper, 100) + 1,
        20
    )
)

ax.tick_params(
    axis='x',
    labelsize=FS_TICK,
    direction='out',
    width=0.6,
    length=2.5,
    pad=3
)

ax.tick_params(
    axis='y',
    labelsize=FS_TICK,
    direction='out',
    width=0.6,
    length=2.5,
    pad=2
)

# Nature guide recommends avoiding background gridlines
ax.grid(False)
ax.set_axisbelow(True)


# =============================================================================
# 8. Full rectangular border
# =============================================================================
for spine in ax.spines.values():
    spine.set_visible(True)
    spine.set_linewidth(0.6)

# =============================================================================
# 10. Legend
# =============================================================================
legend_handles = [
    Patch(
        facecolor=COLOR_EXP_NOBLE,
        edgecolor='black',
        linewidth=0.35,
        label='Experimental (noble-metal)'
    ),
    Patch(
        facecolor=COLOR_PRED_NOBLE,
        edgecolor='black',
        linewidth=0.35,
        label='Predicted (noble-metal)'
    ),
    Patch(
        facecolor=COLOR_EXP_FREE,
        edgecolor='black',
        linewidth=0.35,
        label='Experimental (noble-metal-free)'
    ),
    Patch(
        facecolor=COLOR_PRED_FREE,
        edgecolor='black',
        linewidth=0.35,
        label='Predicted (noble-metal-free)'
    )
]

ax.legend(
    handles=legend_handles,
    loc='lower center',
    bbox_to_anchor=(0.5, 1.08),
    ncol=4,                    # 한 줄로 배치
    frameon=False,
    fontsize=FS_LEGEND,
    handlelength=1.15,
    handleheight=0.8,
    handletextpad=0.35,
    columnspacing=0.90,
    labelspacing=0.0,
    borderaxespad=0
)

fig.subplots_adjust(
    left=0.090,
    right=0.990,
    bottom=0.245,
    top=0.780
)

# plt.savefig('./figure_4a.png', dpi=600, bbox_inches='tight')
# plt.savefig('./figure_4a.pdf', dpi=600, bbox_inches='tight')
save_figure(fig, "figure_04a")  # noqa: F405
