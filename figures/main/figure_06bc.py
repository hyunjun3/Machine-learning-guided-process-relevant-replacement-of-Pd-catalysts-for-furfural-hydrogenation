"""Generate manuscript Figure 6b-c."""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from figures.common import *  # noqa: F403

#%% Fig. 6 - MSP comparison

# ── Data ──────────────────────────────────────────────────────────────────
catalysts = ["Ni–Re", "Pd",     "Ni–Ru", "Ni–Pd", "Ni–Ir",  "Ni–Pt", "Ni–Rh"]
MSP       = np.array([3181.42, 4021.61 , 3439.52, 3873.25, 4523.35, 5175.07, 5366.40])
yield_exp = np.array([84.26,   86.89,   80.45,   73.34,   73.89,   55.48,   66.48])
RMV       = np.array([26.60,  1768.29,  222.44,  354.26, 1415.23,  386.41, 1865.34])
MARKET    = 3500

# ── Normalize x only by mean ──────────────────────────────────────────────
avg_yield = yield_exp.mean()   # ~74.40  → y 기준선으로만 사용
avg_rmv   = RMV.mean()         # ~862.65

rmv_norm  = RMV / avg_rmv      # x축: normalized

# ── Colors / markers ──────────────────────────────────────────────────────
C_PD = "#EE854A"
C_NIBIMETAL = "#74ADD1"
C_NIRE = "#2166AC"
C_MARKET = "#C0392B"

def get_color(catalyst):
    if catalyst == "Ni–Re":
        return C_NIRE
    if catalyst == "Pd":
        return C_PD
    return C_NIBIMETAL

def get_marker(catalyst):
    if catalyst == "Ni–Re":
        return "*"
    if catalyst == "Pd":
        return "o"
    return "s"

colors = [get_color(c) for c in catalysts]
#################################################################################################################
MM_TO_INCH = 1 / 25.4

FIG_WIDTH = 180 * MM_TO_INCH
FIG_HEIGHT = 100 * MM_TO_INCH

FS_PANEL = 8.0
FS_REGION = 6.5
FS_LABEL = 6.5
FS_TICK = 5.5
FS_TEXT = 5.8
FS_LEGEND = 5.8

mpl.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': [
        'Arial',
        'Helvetica',
        'Liberation Sans',
        'DejaVu Sans'
    ],

    'font.size': FS_TICK,
    'axes.labelsize': FS_LABEL,
    'axes.linewidth': 0.6,

    'xtick.labelsize': FS_TICK,
    'ytick.labelsize': FS_TICK,

    'xtick.major.size': 2.5,
    'ytick.major.size': 2.5,
    'xtick.major.width': 0.6,
    'ytick.major.width': 0.6,

    'xtick.direction': 'out',
    'ytick.direction': 'out',

    'pdf.fonttype': 42,
    'ps.fonttype': 42,
    'svg.fonttype': 'none',

    'savefig.facecolor': 'white'
})


# =============================================================================
# 2. Data
# 실제 분석값으로 교체
# =============================================================================
plot_df = pd.DataFrame({
    'Catalyst': [
        'Ni–Re',
        'Ni–Ru',
        'Ni–Pd',
        'Ni–Pt',
        'Ni–Ir',
        'Ni–Rh',
        'Pd'
    ],

    'Experimental_yield': [
        84.3,
        80.5,
        73.4,
        55.5,
        73.8,
        66.5,
        86.8
    ],

    'Relative_cost': [
        0.031,
        0.255,
        0.410,
        0.450,
        1.65,
        2.15,
        2.02
    ],

       'MSP': [
        3181.4,
        3439.5,
        3873.3,
        5175.1,
        4523.4,
        5366.4,
        4021.61
    ],

    'Type': [
        'Candidate',
        'Ni-based',
        'Ni-based',
        'Ni-based',
        'Ni-based',
        'Ni-based',
        'Pd'
    ]
})


# =============================================================================
# 3. Thresholds
# =============================================================================
# Panel b
mean_yield = 74.3
mean_relative_cost = 1.0

# Panel c
high_yield_threshold = 80.0
market_price_low = 3000
market_price_high = 5000

# =============================================================================
# 4. Colors
# =============================================================================
COLOR_CANDIDATE = '#2C6DB2'
COLOR_NI = '#6BAED6'
COLOR_PD = '#F28E5B'

COLOR_TARGET_GREEN = '#5B8C5A'
COLOR_HIGH_COST = '#9A6A36'
COLOR_LOW_COST = '#536D8D'
COLOR_POOR = '#8A5A5A'

COLOR_TARGET_BLUE = '#2C6DB2'
COLOR_MARKET = '#C43C35'

COLOR_REFERENCE = '0.45'
COLOR_EDGE = '0.25'


# =============================================================================
# 5. Figure layout
# =============================================================================
fig, axes = plt.subplots(
    nrows=1,
    ncols=2,
    figsize=(FIG_WIDTH, FIG_HEIGHT),
    dpi=300
)

ax_b, ax_c = axes

fig.subplots_adjust(
    left=0.085,
    right=0.985,
    bottom=0.225,
    top=0.930,
    wspace=0.28
)


# =============================================================================
# 6. Common plotting function
# =============================================================================
def draw_catalyst_markers(ax, x_column):
    """
    Ni–Re candidate, Ni-based catalysts, and Pd benchmark를
    서로 다른 marker로 표시합니다.
    """

    ni_data = plot_df[
        plot_df['Type'] == 'Ni-based'
    ]

    candidate_data = plot_df[
        plot_df['Type'] == 'Candidate'
    ]

    pd_data = plot_df[
        plot_df['Type'] == 'Pd'
    ]

    # Ni-based bimetallic catalysts
    ax.scatter(
        ni_data[x_column],
        ni_data['Experimental_yield'],
        marker='s',
        s=42,
        facecolor=COLOR_NI,
        edgecolor=COLOR_EDGE,
        linewidth=0.55,
        zorder=4
    )

    # Ni–Re candidate
    ax.scatter(
        candidate_data[x_column],
        candidate_data['Experimental_yield'],
        marker='*',
        s=125,
        facecolor=COLOR_CANDIDATE,
        edgecolor=COLOR_EDGE,
        linewidth=0.65,
        zorder=5
    )

    # Pd benchmark
    ax.scatter(
        pd_data[x_column],
        pd_data['Experimental_yield'],
        marker='o',
        s=52,
        facecolor=COLOR_PD,
        edgecolor=COLOR_EDGE,
        linewidth=0.60,
        zorder=5
    )


def style_axis(ax):
    """
    공통 축 스타일.
    """

    ax.grid(False)

    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(0.6)

    ax.tick_params(
        axis='both',
        which='major',
        direction='out',
        width=0.6,
        length=2.5,
        pad=2
    )

    ax.set_axisbelow(True)


# =============================================================================
# 7. Panel b: Experimental yield vs relative catalyst cost
# =============================================================================
b_xmin = 0.015
b_xmax = 3.5
b_ymin = 48
b_ymax = 95


# -----------------------------------------------------------------------------
# 7.1 Decision-region shading
# -----------------------------------------------------------------------------
# High yield / low cost: 핵심 목표 영역
ax_b.add_patch(
    Rectangle(
        (b_xmin, mean_yield),
        mean_relative_cost - b_xmin,
        b_ymax - mean_yield,
        facecolor=COLOR_TARGET_GREEN,
        edgecolor='none',
        alpha=0.085,
        zorder=0
    )
)

# High yield / high cost
ax_b.add_patch(
    Rectangle(
        (mean_relative_cost, mean_yield),
        b_xmax - mean_relative_cost,
        b_ymax - mean_yield,
        facecolor=COLOR_HIGH_COST,
        edgecolor='none',
        alpha=0.030,
        zorder=0
    )
)

# Low yield / low cost
ax_b.add_patch(
    Rectangle(
        (b_xmin, b_ymin),
        mean_relative_cost - b_xmin,
        mean_yield - b_ymin,
        facecolor=COLOR_LOW_COST,
        edgecolor='none',
        alpha=0.020,
        zorder=0
    )
)

# Low yield / high cost
ax_b.add_patch(
    Rectangle(
        (mean_relative_cost, b_ymin),
        b_xmax - mean_relative_cost,
        mean_yield - b_ymin,
        facecolor=COLOR_POOR,
        edgecolor='none',
        alpha=0.025,
        zorder=0
    )
)


# -----------------------------------------------------------------------------
# 7.2 Mean reference lines
# -----------------------------------------------------------------------------
ax_b.axhline(
    mean_yield,
    color=COLOR_REFERENCE,
    linestyle='--',
    linewidth=0.8,
    zorder=2
)

ax_b.axvline(
    mean_relative_cost,
    color=COLOR_REFERENCE,
    linestyle='--',
    linewidth=0.8,
    zorder=2
)


# -----------------------------------------------------------------------------
# 7.3 Catalyst markers
# -----------------------------------------------------------------------------
draw_catalyst_markers(
    ax=ax_b,
    x_column='Relative_cost'
)


# -----------------------------------------------------------------------------
# 7.4 Catalyst labels
# -----------------------------------------------------------------------------
label_offsets_b = {
    'Ni–Re': (0, 9),
    'Ni–Ru': (-1, 8),
    'Ni–Pd': (0, -9),
    'Ni–Pt': (0, -9),
    'Ni–Ir': (0, 8),
    'Ni–Rh': (0, 8),
    'Pd': (-0.5, 8)
}

for _, row in plot_df.iterrows():

    catalyst = row['Catalyst']
    dx, dy = label_offsets_b[catalyst]

    ax_b.annotate(
        catalyst,
        xy=(
            row['Relative_cost'],
            row['Experimental_yield']
        ),
        xytext=(dx, dy),
        textcoords='offset points',
        ha='center',
        va='center',
        fontsize=FS_TEXT,
        color='black',
        zorder=6
    )


# -----------------------------------------------------------------------------
# 7.5 Region labels
# -----------------------------------------------------------------------------
ax_b.text(
    0.018,
    93.5,
    'High yield / low cost',
    fontsize=FS_REGION,
    fontweight='bold',
    color='#496F49',
    ha='left',
    va='top'
)

ax_b.text(
    3.15,
    94.0,
    'High yield /\nhigh cost',
    fontsize=FS_REGION,
    fontweight='bold',
    color='#7A582C',
    ha='right',
    va='top',
    linespacing=0.95
)

ax_b.text(
    0.018,
    49.5,
    'Low yield / low cost',
    fontsize=FS_REGION,
    fontweight='bold',
    color='#4C6482',
    ha='left',
    va='bottom'
)

ax_b.text(
    3.15,
    49.5,
    'Low yield /\nhigh cost',
    fontsize=FS_REGION,
    fontweight='bold',
    color='#805454',
    ha='right',
    va='bottom',
    linespacing=0.95
)


# -----------------------------------------------------------------------------
# 7.6 Reference labels
# -----------------------------------------------------------------------------
ax_b.text(
    0.017,
    mean_yield + 0.7,
    'Mean yield',
    fontsize=FS_TEXT,
    color=COLOR_REFERENCE,
    ha='left',
    va='bottom'
)

ax_b.text(
    mean_relative_cost * 1.035,
    51.0,
    'Mean cost',
    fontsize=FS_TEXT,
    color=COLOR_REFERENCE,
    rotation=90,
    ha='left',
    va='bottom'
)


# -----------------------------------------------------------------------------
# 7.7 Axes
# -----------------------------------------------------------------------------
ax_b.set_xscale('log')

ax_b.set_xlim(
    b_xmin,
    b_xmax
)

ax_b.set_ylim(
    b_ymin,
    b_ymax
)

ax_b.set_xticks(
    [0.03, 0.1, 0.3, 1, 3]
)

ax_b.set_xticklabels(
    ['0.03', '0.1', '0.3', '1', '3']
)

ax_b.set_yticks(
    [50, 60, 70, 80, 90]
)

ax_b.set_xlabel(
    'Relative catalyst raw-material cost',
    fontsize=FS_LABEL,
    labelpad=4
)

ax_b.set_ylabel(
    'Experimental THFA yield (%)',
    fontsize=FS_LABEL,
    labelpad=4
)

style_axis(ax_b)


# =============================================================================
# 8. Panel c: Experimental yield vs MSP
# =============================================================================
c_xmin = 2900
c_xmax = 5700
c_ymin = 48
c_ymax = 95


# -----------------------------------------------------------------------------
# 8.1 Target region
# -----------------------------------------------------------------------------
ax_c.axvspan(
    market_price_low,
    market_price_high,
    facecolor=COLOR_MARKET,
    edgecolor='none',
    alpha=0.055,
    zorder=0
)


# -----------------------------------------------------------------------------
# 8.2 Market-price reference
# -----------------------------------------------------------------------------
for x_bound in (market_price_low, market_price_high):
    ax_c.axvline(
        x_bound,
        color=COLOR_MARKET,
        linestyle='--',
        linewidth=0.8,
        alpha=0.85,
        zorder=2
    )
# -----------------------------------------------------------------------------
# 8.3 Catalyst markers
# -----------------------------------------------------------------------------
draw_catalyst_markers(
    ax=ax_c,
    x_column='MSP'
)


# -----------------------------------------------------------------------------
# 8.4 Catalyst labels and leader lines
# -----------------------------------------------------------------------------
label_offsets_c = {
    'Ni–Re': (0, 10),
    'Ni–Ru': (0, 8),
    'Ni–Pd': (0, -9),
    'Ni–Pt': (0, 8),
    'Ni–Ir': (0, 8),
    'Ni–Rh': (0, 8),
    'Pd': (0, 8)
}

leader_line_catalysts = set()

for _, row in plot_df.iterrows():

    catalyst = row['Catalyst']
    dx, dy = label_offsets_c[catalyst]

    arrowprops = None

    if catalyst in leader_line_catalysts:
        arrowprops = {
            'arrowstyle': '-',
            'color': '0.55',
            'linewidth': 0.55,
            'shrinkA': 1,
            'shrinkB': 3
        }

    ax_c.annotate(
        catalyst,
        xy=(
            row['MSP'],
            row['Experimental_yield']
        ),
        xytext=(dx, dy),
        textcoords='offset points',
        ha='center',
        va='center',
        fontsize=FS_TEXT,
        color='black',
        arrowprops=arrowprops,
        zorder=6
    )


# -----------------------------------------------------------------------------
# 8.5 Target-region and market-price labels
# -----------------------------------------------------------------------------
ax_c.axhline(
    high_yield_threshold,
    color=COLOR_REFERENCE,
    linestyle='--',
    linewidth=0.8,
    zorder=2
)
ax_c.text(
    (market_price_low + market_price_high) / 2,
    93.5,
    'THFA market price range (3,000–5,000 $/ton)',
    fontsize=FS_REGION,
    fontweight='bold',
    color=COLOR_MARKET,
    ha='center',
    va='top'
)


# -----------------------------------------------------------------------------
# 8.6 Axes
# -----------------------------------------------------------------------------
ax_c.set_xlim(
    c_xmin,
    c_xmax
)

ax_c.set_ylim(
    c_ymin,
    c_ymax
)

ax_c.set_xticks(
    [3000, 3500, 4000, 4500, 5000, 5500]
)

ax_c.set_yticks(
    [50, 60, 70, 80, 90]
)

ax_c.set_xlabel(
    'Minimum selling price ($/ton THFA)',
    fontsize=FS_LABEL,
    labelpad=4
)

ax_c.set_ylabel(
    'Experimental THFA yield (%)',
    fontsize=FS_LABEL,
    labelpad=4
)

style_axis(ax_c)


# =============================================================================
# 9. Panel labels
# =============================================================================
ax_b.text(
    0.00,
    1.02,
    'b',
    transform=ax_b.transAxes,
    fontsize=FS_PANEL,
    fontweight='bold',
    ha='left',
    va='bottom',
    clip_on=False
)

ax_c.text(
    0.00,
    1.02,
    'c',
    transform=ax_c.transAxes,
    fontsize=FS_PANEL,
    fontweight='bold',
    ha='left',
    va='bottom',
    clip_on=False
)


# =============================================================================
# 10. Shared legend
# =============================================================================
legend_handles = [
    Line2D(
        [0],
        [0],
        marker='*',
        linestyle='none',
        markersize=8.5,
        markerfacecolor=COLOR_CANDIDATE,
        markeredgecolor=COLOR_EDGE,
        markeredgewidth=0.6,
        label='Ni–Re candidate'
    ),

    Line2D(
        [0],
        [0],
        marker='s',
        linestyle='none',
        markersize=5.5,
        markerfacecolor=COLOR_NI,
        markeredgecolor=COLOR_EDGE,
        markeredgewidth=0.6,
        label='Ni-based bimetallic catalysts'
    ),

    Line2D(
        [0],
        [0],
        marker='o',
        linestyle='none',
        markersize=5.8,
        markerfacecolor=COLOR_PD,
        markeredgecolor=COLOR_EDGE,
        markeredgewidth=0.6,
        label='Pd benchmark'
    )
]

fig.legend(
    handles=legend_handles,
    loc='lower center',
    bbox_to_anchor=(0.5, 0.095),
    ncol=3,
    frameon=False,
    fontsize=FS_LEGEND,
    handlelength=1.0,
    handletextpad=0.45,
    columnspacing=1.6,
    borderaxespad=0
)

# plt.savefig('./figure_6_bc.png', dpi=600, bbox_inches='tight')
# plt.savefig('./figure_6_bc.pdf', dpi=600, bbox_inches='tight') 
# fig.savefig('./figure_6_bc.svg')   
save_figure(fig, "figure_06bc")  # noqa: F405
