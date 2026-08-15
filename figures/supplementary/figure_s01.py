"""Generate Supplementary Figure 1."""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from figures.common import *  # noqa: F403

globals().update(load_ml_context())  # noqa: F405


#%% Supplementary figures - Fig. 1 (Dataset distribution)

active_counts = (dataset[active_metal].gt(0).sum().sort_values(ascending=False))
active_counts = active_counts[active_counts > 0]

support_counts = (dataset[cat_support].gt(0).sum().sort_values(ascending=False))
support_counts = support_counts[support_counts > 0]

precursor_counts = ( dataset[precursor].gt(0).sum().sort_values(ascending=False))
precursor_counts = precursor_counts[precursor_counts > 0]

precursor_counts = precursor_counts.rename(index={'Unknown_precursor': 'Unknown precursor'})


preparation_counts = (dataset[preparation].gt(0).sum().sort_values(ascending=False))
preparation_counts = preparation_counts[preparation_counts > 0]

preparation_counts = preparation_counts.rename(index={'Unknown_preparation': 'Unknown preparation'})

solvent_counts = (dataset[solvent].gt(0).sum().sort_values(ascending=False))
solvent_counts = solvent_counts[solvent_counts > 0]

yield_col = 'THFA_yield (%)'
yield_data = dataset[yield_col].dropna().astype(float)



MM_TO_INCH = 1 / 25.4

FIG_WIDTH = 180 * MM_TO_INCH
FIG_HEIGHT = 150 * MM_TO_INCH

FS_PANEL = 8.0
FS_TITLE = 7.0
FS_LABEL = 6.5
FS_TICK = 6.0
FS_SMALL = 5.5

mpl.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': 'Arial',

    'font.size': FS_TICK,

    'axes.labelsize': FS_LABEL,
    'axes.titlesize': FS_TITLE,
    'axes.linewidth': 0.6,

    'xtick.labelsize': FS_TICK,
    'ytick.labelsize': FS_SMALL,
    'xtick.major.size': 2.5,
    'ytick.major.size': 2.5,
    'xtick.major.width': 0.6,
    'ytick.major.width': 0.6,
    'xtick.direction': 'out',
    'ytick.direction': 'out',

    'lines.linewidth': 0.9,

    # PDF에서 텍스트 편집 가능
    'pdf.fonttype': 42,
    'ps.fonttype': 42,
    'svg.fonttype': 'none',

    'savefig.facecolor': 'white'
})

# =============================================================================
# 3. Plotting helpers
# =============================================================================
def set_panel_header(ax, panel_label, title):
    """
    패널 문자와 패널 제목을 동일한 높이에 배치합니다.
    """

    ax.text(
        0.00,
        1.045,
        panel_label,
        transform=ax.transAxes,
        ha='left',
        va='bottom',
        fontsize=FS_PANEL,
        fontweight='bold',
        fontstyle='normal',
        color='black',
        clip_on=False
    )

    # Subplot title
    ax.text(
        0.105,
        1.045,
        title,
        transform=ax.transAxes,
        ha='left',
        va='bottom',
        fontsize=FS_TITLE,
        fontweight='normal',
        fontstyle='normal',
        color='black',
        clip_on=False
    )


def style_axis(ax, grid_axis=None):
    """
    모든 패널에 공통적인 NCE 스타일을 적용합니다.
    """

    # Full rectangular box
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(0.6)

    ax.tick_params(
        axis='both',
        which='major',
        direction='out',
        width=0.6,
        length=2.5
    )

    # Background grid 제거
    ax.grid(False)

    ax.set_axisbelow(True)


def plot_bar(
    ax,
    series,
    panel_label,
    title,
    color,
    top_n=15
):
    """
    범주별 데이터 개수를 나타내는 수평 막대그래프입니다.
    """

    s = series.copy()

    if top_n is not None:
        s = s.head(top_n)

    # barh에서 가장 큰 값이 위쪽에 오도록 오름차순으로 재정렬
    s = s.sort_values(ascending=True)

    ax.barh(
        y=s.index,
        width=s.values,
        height=0.72,
        color=color,
        edgecolor='none',
        zorder=2
    )

    set_panel_header(
        ax=ax,
        panel_label=panel_label,
        title=title
    )

    ax.set_xlabel('Count', fontsize=FS_LABEL)
    ax.set_ylabel('')

    ax.tick_params(
        axis='x',
        labelsize=FS_TICK,
        pad=2
    )

    ax.tick_params(
        axis='y',
        labelsize=FS_SMALL,
        length=0,
        pad=2
    )

    # Count 축은 정수 눈금 사용
    ax.xaxis.set_major_locator(
        MaxNLocator(
            nbins=5,
            integer=True,
            min_n_ticks=3
        )
    )

    ax.margins(y=0.025)

    style_axis(
        ax,
        grid_axis='x'
    )


# =============================================================================
# 4. Figure layout
# =============================================================================
fig = plt.figure(
    figsize=(FIG_WIDTH, FIG_HEIGHT),
    dpi=300
)

gs = fig.add_gridspec(
    nrows=2,
    ncols=3,

    left=0.075,
    right=0.990,
    bottom=0.085,
    top=0.955,

    # 긴 범주명이 옆 패널과 겹치지 않도록 확보
    wspace=0.60,
    hspace=0.35
)

axes = np.array([
    [fig.add_subplot(gs[0, 0]),
     fig.add_subplot(gs[0, 1]),
     fig.add_subplot(gs[0, 2])],

    [fig.add_subplot(gs[1, 0]),
     fig.add_subplot(gs[1, 1]),
     fig.add_subplot(gs[1, 2])]
])


# =============================================================================
# 5. Categorical statistics
# =============================================================================
plot_bar(
    axes[0, 0],
    active_counts,
    panel_label='a',
    title='',
    color='#4C78A8',
    top_n=15
)

plot_bar(
    axes[0, 1],
    precursor_counts,
    panel_label='b',
    title='',
    color='#54A24B',
    top_n=15
)

plot_bar(
    axes[0, 2],
    support_counts,
    panel_label='c',
    title='',
    color='#F58518',
    top_n=15
)

plot_bar(
    axes[1, 0],
    preparation_counts,
    panel_label='d',
    title='',
    color='#E45756',
    top_n=15
)

plot_bar(
    axes[1, 1],
    solvent_counts,
    panel_label='e',
    title='',
    color='#B279A2',
    top_n=15
)


# =============================================================================
# 6. THFA yield distribution
# =============================================================================
ax_yield = axes[1, 2]

sns.histplot(
    data=yield_data,
    bins=np.linspace(0, 100, 21),
    kde=True,
    stat='count',
    ax=ax_yield,

    color='#72B7B2',
    edgecolor='black',
    linewidth=0.35,
    alpha=0.75,

    line_kws={
        'linewidth': 1.0
    }
)

set_panel_header(
    ax=ax_yield,
    panel_label='f',
    title=''
)

ax_yield.set_xlim(0, 100)

ax_yield.set_xlabel(
    'THFA yield (%)',
    fontsize=FS_LABEL
)

ax_yield.set_ylabel(
    'Frequency',
    fontsize=FS_LABEL
)

ax_yield.tick_params(
    axis='both',
    labelsize=FS_TICK
)

ax_yield.xaxis.set_major_locator(
    MaxNLocator(
        nbins=6,
        integer=True
    )
)

ax_yield.yaxis.set_major_locator(
    MaxNLocator(
        nbins=5,
        integer=True
    )
)

style_axis(
    ax_yield,
    grid_axis='y'
)

# plt.savefig('./figure_supp_1.png', dpi=600, bbox_inches='tight')
# plt.savefig('./figure_supp_1.pdf', dpi=600, bbox_inches='tight')
save_figure(fig, "figure_s01")  # noqa: F405
