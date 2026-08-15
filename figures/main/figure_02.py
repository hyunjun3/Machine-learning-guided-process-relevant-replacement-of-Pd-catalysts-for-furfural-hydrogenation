"""Generate manuscript Figure 2."""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from figures.common import *  # noqa: F403

globals().update(load_ml_context())  # noqa: F405


#%% Fig. 2 ML predictive basis / SHAP / global PDP

MM_TO_INCH = 1 / 25.4

FIG_WIDTH = 180 * MM_TO_INCH
FIG_HEIGHT = 245 * MM_TO_INCH

FS_PANEL = 8.0
FS_LABEL = 7.0
FS_TICK = 6.0
FS_TEXT = 6.0
FS_SMALL = 5.5
FS_CBAR = 6.0
PANEL_PAD = 5

mpl.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': 'Arial',

    'font.size': FS_TEXT,

    'axes.labelsize': FS_LABEL,
    'axes.titlesize': FS_PANEL,
    'axes.linewidth': 0.6,

    'xtick.labelsize': FS_TICK,
    'ytick.labelsize': FS_TICK,

    'xtick.major.size': 2.5,
    'ytick.major.size': 2.5,
    'xtick.major.width': 0.6,
    'ytick.major.width': 0.6,

    'xtick.direction': 'out',
    'ytick.direction': 'out',

    'lines.linewidth': 0.9,
    'lines.markersize': 3.5,

    'legend.fontsize': FS_SMALL,
    'legend.frameon': False,

    'pdf.fonttype': 42,
    'ps.fonttype': 42,
    'svg.fonttype': 'none',

    'savefig.facecolor': 'white'
})


def draw_2d_pdv(
    fig,
    subspec,
    model,
    X_train,
    prop_1,
    prop_2,
    xlim=None,
    ylim=None,
    x_ticks=None,
    y_ticks=None,
    x_label='',
    y_label='',
    title='',
    grid_resolution=30,
    cmap='RdYlBu_r',
    zmin=0,
    zmax=100
):

    X_pdp = X_train.astype(np.float64)
    X_custom = X_pdp.copy()

    if prop_1 == 'Calcination_temp':
        X_custom = X_custom[X_custom[prop_1] >= 100]

    if prop_2 == 'Calcination_time':
        X_custom = X_custom[X_custom[prop_2] <= 10]

    if prop_1 == 'Reduction_temp':
        X_custom = X_custom[X_custom[prop_1] >= 100]

    if prop_2 == 'Reduction_time':
        X_custom = X_custom[X_custom[prop_2] <= 10]

    if prop_1 == 'Operating_temp':
        X_custom = X_custom[X_custom[prop_1] >= 50]

    if prop_2 == 'Operating_time':
        X_custom = X_custom[X_custom[prop_2] <= 10]

    X_custom = X_custom.copy()

    X_custom[prop_1] = pd.to_numeric(X_custom[prop_1], errors='coerce')

    X_custom[prop_2] = pd.to_numeric(X_custom[prop_2], errors='coerce')

    X_custom = X_custom.dropna(subset=[prop_1, prop_2])

    custom_grid = {
        prop_1: np.linspace(
            X_custom[prop_1].min(),
            X_custom[prop_1].max(),
            grid_resolution
        ),
        prop_2: np.linspace(
            X_custom[prop_2].min(),
            X_custom[prop_2].max(),
            grid_resolution
        )
    }

    if (prop_1 == 'Operating_temp' and prop_2 == 'Operating_time'):
        custom_grid = {
            prop_1: np.linspace(
                50,
                X_custom[prop_1].max(),
                grid_resolution
            ),
            prop_2: np.linspace(
                X_custom[prop_2].min(),
                10,
                grid_resolution
            )
        }

    if (prop_1 == 'Reduction_temp' and prop_2 == 'Reduction_time'):
        custom_grid = {
            prop_1: np.linspace(
                100,
                X_custom[prop_1].max(),
                grid_resolution
            ),
            prop_2: np.linspace(
                X_custom[prop_2].min(),
                10,
                grid_resolution
            )
        }

    if (prop_1 == 'Calcination_temp' and prop_2 == 'Calcination_time'):
        custom_grid = {
            prop_1: np.linspace(
                100,
                X_custom[prop_1].max(),
                grid_resolution
            ),
            prop_2: np.linspace(
                X_custom[prop_2].min(),
                10,
                grid_resolution
            )
        }

    #PDV calculation
    pd_results = partial_dependence(
        model,
        X_pdp,
        features=[prop_1, prop_2],
        custom_values=custom_grid
    )

    x_axis = np.asarray(
        pd_results['grid_values'][0],
        dtype=float
    )

    y_axis = np.asarray(
        pd_results['grid_values'][1],
        dtype=float
    )

    Xg, Yg = np.meshgrid(
        x_axis,
        y_axis,
        indexing='ij'
    )

    Z_raw = np.asarray(
        pd_results['average'][0],
        dtype=float
    ).reshape(
        len(x_axis),
        len(y_axis)
    )

    #Colorbar setting
    clip_zmin = (-np.inf if zmin is None else zmin)
    clip_zmax = (np.inf if zmax is None else zmax)

    Z = np.clip(Z_raw, clip_zmin, clip_zmax)
    
    #set axes
    gs = subspec.subgridspec(2, 3,
        width_ratios=[1.0, 5.0, 0.24], height_ratios=[1.0, 5.0],
        wspace=0.06, hspace=0.06)

    ax_joint = fig.add_subplot(gs[1, 1])
    ax_marg_x = fig.add_subplot(gs[0, 1], sharex=ax_joint)
    ax_marg_y = fig.add_subplot(gs[1, 0], sharey=ax_joint)

    ax_cb = fig.add_subplot(gs[1, 2])

    cp = ax_joint.contourf(Xg, Yg, Z, levels=40, cmap=cmap, antialiased=False)

    for collection in getattr(
        cp,
        'collections',
        []
    ):
        collection.set_edgecolor('face')
        collection.set_linewidth(0.0)
        collection.set_antialiased(False)
        collection.set_rasterized(True)

    if xlim is not None:
        ax_joint.set_xlim(*xlim)
        ax_marg_x.set_xlim(*xlim)

    if ylim is not None:
        ax_joint.set_ylim(*ylim)
        ax_marg_y.set_ylim(*ylim)

    sns.histplot(
        data=X_custom,
        x=prop_1,
        ax=ax_marg_x,
        color='gray',
        kde=True,
        alpha=0.28,
        bins=20,
        element='step',
        linewidth=0.7
    )

    ax_marg_x.set_xlabel('')

    ax_marg_x.set_ylabel('Density', fontsize=FS_SMALL, labelpad=1)

    ax_marg_x.tick_params(
        axis='x',
        which='both',
        bottom=False,
        top=False,
        labelbottom=False,
        labeltop=False
    )

    ax_marg_x.set_yticks([])

    sns.histplot(
        data=X_custom,
        y=prop_2,
        ax=ax_marg_y,
        color='gray',
        kde=True,
        alpha=0.28,
        bins=15,
        element='step',
        linewidth=0.7
    )

    ax_marg_y.invert_xaxis()

    ax_marg_y.set_xlabel('Density', fontsize=FS_SMALL, labelpad=1)

    ax_marg_y.set_ylabel(y_label, fontsize=FS_LABEL, labelpad=7)

    ax_marg_y.yaxis.set_label_position('left')

    ax_marg_y.yaxis.tick_left()
    ax_marg_y.set_xticks([])


    if x_ticks is not None:
        ax_joint.set_xticks(x_ticks)

    if y_ticks is not None:
        ax_marg_y.set_yticks(y_ticks)

    ax_joint.set_xlabel(x_label, fontsize=FS_LABEL)
    ax_joint.set_ylabel('')

    ax_joint.tick_params(
        axis='x',
        labelsize=FS_TICK,
        width=0.6,
        length=2.5,
        direction='out'
    )

    ax_joint.tick_params(
        axis='y',
        which='both',
        left=False,
        labelleft=False
    )

    ax_marg_y.tick_params(
        axis='y',
        labelsize=FS_TICK,
        width=0.6,
        length=2.5,
        direction='out'
    )

    ax_joint.set_aspect(
        'auto'
    )

    for current_ax in [
        ax_joint,
        ax_marg_x,
        ax_marg_y
    ]:
        for spine in current_ax.spines.values():
            spine.set_visible(True)
            spine.set_linewidth(0.6)

        current_ax.grid(False)

    cbar = fig.colorbar(cp, cax=ax_cb)
    cbar.set_label('PDV', fontsize=FS_CBAR, labelpad=3)
    cbar.ax.tick_params(labelsize=FS_SMALL, width=0.6, length=2, direction='out')
    cbar.outline.set_linewidth(0.6)

    return {
    'joint': ax_joint,
    'marg_x': ax_marg_x,
    'marg_y': ax_marg_y,
    'colorbar_ax': ax_cb,
    'contour': cp
    }

def draw_1d_pdv(ax, model, X_train, feature,  xlabel, title, color, xlim=None, x_ticks=None):  
    X_pdp = X_train.astype(np.float64)
    
    display = PartialDependenceDisplay.from_estimator(
        model, X_pdp, features=[feature], kind='average',
        centered=False, grid_resolution=50,  ax=ax,
        line_kw={'color': color, 'linewidth': 1.2})

    real_ax = display.axes_[0, 0]

    if xlim is not None:
        real_ax.set_xlim(*xlim)
    if x_ticks is not None:
        real_ax.set_xticks(x_ticks)

    real_ax.tick_params(
        axis='both',
        which='major',
        labelsize=FS_TICK,
        direction='out',
        width=0.6,
        length=2.5,
        pad=2
    )

    real_ax.set_xlabel(xlabel, fontsize=FS_LABEL, labelpad=3)

    real_ax.set_ylabel('PDV', fontsize=FS_LABEL, labelpad=3)

    real_ax.grid(False)
    real_ax.xaxis.grid(False)
    real_ax.yaxis.grid(False)

    for spine in real_ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(0.6)

    return real_ax

fig = plt.figure(figsize=(FIG_WIDTH, FIG_HEIGHT), dpi=300)

outer = fig.add_gridspec(3, 2, width_ratios=[1.0, 1.0], height_ratios=[1.2, 1.0, 1.0], wspace=0.28, hspace=0.20)

#Parity plot
ax1 = fig.add_subplot(outer[0, 0])
scatter = ax1.scatter(
    y_test,
    y_test_pred,
    c='#56B4E9',
    edgecolors='white',
    linewidths=0.45,
    s=22,
    alpha=0.85,
    zorder=3,
    label='Test set'
)

lims = [-5, 105]

ax1.plot(lims, lims, color='black', linewidth=0.9, linestyle='--', zorder=2)

ax1.fill_between(
    lims,
    [
        value - 10
        for value in lims
    ],
    [
        value + 10
        for value in lims
    ],
    color='gray',
    alpha=0.12,
    zorder=1,
    label='±10%'
)

textstr = (
    f'$R^2$ = {XGB_test_r2:.4f}\n'
    f'RMSE = {XGB_test_rmse:.2f}'
)

ax1.text(
    0.05,
    0.95,
    textstr,
    transform=ax1.transAxes,
    fontsize=FS_TEXT,
    verticalalignment='top',
    bbox={
        'boxstyle': 'round,pad=0.3',
        'facecolor': 'white',
        'edgecolor': '#CCCCCC',
        'linewidth': 0.6,
        'alpha': 0.9
    }
)

ax1.set_xlim(-5, 105)
ax1.set_ylim(-5, 105)
ax1.set_xlabel('Actual THFA yield (%)', fontsize=FS_LABEL)
ax1.set_ylabel('Predicted THFA yield (%)', fontsize=FS_LABEL)
ax1.set_aspect('equal', adjustable='box')
ax1.set_anchor('C')

for spine in ['top', 'right']:
    ax1.spines[spine].set_visible(False)

ax1.spines['left'].set_linewidth(0.6)
ax1.spines['bottom'].set_linewidth(0.6)

ax1.tick_params(axis='both', which='major', labelsize=FS_TICK, direction='out', length=2.5, width=0.6)


#SHAP summary plot
shap_grid = outer[0, 1].subgridspec(1, 2, width_ratios=[0.16, 1.0], wspace=0.0)
ax2_margin = fig.add_subplot(shap_grid[0, 0])
ax2_margin.axis('off')
ax2 = fig.add_subplot(shap_grid[0, 1])
axes_before_shap = set(fig.axes)
plt.sca(ax2)

rename_dict = {
    'Operating_temp': 'Operating temperature',
    'Operating_time': 'Operating time',
    'Operating_pressure': 'Operating pressure',
    'Furfural (mg)': 'Furfural amount',
    'Active metal_Ni': 'Ni',
    'Substrate concentration (mg/ml)': 'Substrate concentration',
    'Substrate to metal ratio (mmol/mmol)': 'Substrate-to-metal ratio',
    'Stirring rate (rpm)': 'Stirring rate',
    'Reduction_temp': 'Reduction temperature',
    'Reduction_time': 'Reduction time',
    'Calcination_temp': 'Calcination temperature',
    'Calcination_time': 'Calcination time',
    'water': 'Water',
    'ethanol': 'Ethanol'}

X_train_renamed = X_train.rename(columns=rename_dict)
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X_train_renamed)
mean_abs_shap = (np.abs(shap_values).mean(axis=0))

shap_df = pd.DataFrame({'feature': X_train_renamed.columns,
                        'mean_abs_shap': mean_abs_shap}).sort_values('mean_abs_shap', ascending=False)

shap.summary_plot(shap_values, X_train_renamed, max_display=15, show=False, plot_size=None)

for line in list(ax2.lines):

    xdata = np.asarray(line.get_xdata(), dtype=float)
    ydata = np.asarray(line.get_ydata(), dtype=float)

    is_horizontal_line = (ydata.size >= 2 and np.allclose(ydata, ydata[0]) and not np.allclose(xdata, xdata[0]))

    if is_horizontal_line:
        line.remove()

ax2.grid(False)
ax2.xaxis.grid(False)
ax2.yaxis.grid(False)

ax2.set_xlabel('SHAP value (impact on model output)', fontsize=FS_LABEL)
ax2.tick_params(axis='x', labelsize=FS_TICK, width=0.6, length=2.5, direction='out')
ax2.tick_params(axis='y', labelsize=FS_SMALL, width=0.6, length=0, pad=2)

for label in ax2.get_yticklabels():
    label.set_fontsize(FS_SMALL)
    label.set_fontweight('normal')
    label.set_horizontalalignment('right')

fig.canvas.draw()
new_shap_axes = [
    current_ax
    for current_ax in fig.axes
    if (current_ax not in axes_before_shap and current_ax is not ax2)]

if new_shap_axes:
    shap_cbar_ax = new_shap_axes[-1]

    shap_cbar_ax.set_ylabel('Feature value', fontsize=FS_LABEL, labelpad=4)
    shap_cbar_ax.tick_params(labelsize=FS_SMALL, width=0.6, length=2)

#2D PDP plot
panel_c = draw_2d_pdv(
    fig,
    outer[1, 0],
    model,
    X_train,
    prop_1='Calcination_temp',
    prop_2='Calcination_time',
    xlim=(100, 800),
    ylim=(1, 8),
    x_ticks=np.arange(100, 801, 100),
    y_ticks=np.arange(1, 9, 1),
    x_label='Calcination temperature (°C)',
    y_label='Calcination time (h)',
    title='',
    cmap='RdYlBu_r'
)

panel_d = draw_2d_pdv(
    fig,
    outer[1, 1],
    model,
    X_train,
    prop_1='Reduction_temp',
    prop_2='Reduction_time',
    xlim=(100, 800),
    ylim=(1, 8),
    x_ticks=np.arange(100, 801, 100),
    y_ticks=np.arange(1, 9, 1),
    x_label='Reduction temperature (°C)',
    y_label='Reduction time (h)',
    title='',
    cmap='RdYlBu_r'
)

panel_e = draw_2d_pdv(
    fig,
    outer[2, 0],
    model,
    X_train,
    prop_1='Operating_temp',
    prop_2='Operating_time',
    xlim=(50, 250),
    ylim=(1, 10),
    x_ticks=np.arange(50, 251, 50),
    y_ticks=np.arange(1, 11, 1),
    x_label='Operating temperature (°C)',
    y_label='Operating time (h)',
    title='',
    cmap='RdYlBu_r'
)

#1D PDP plot
inner = outer[2, 1].subgridspec(2, 1, height_ratios=[1.0, 1.0], hspace=0.34)
ax_f = fig.add_subplot(inner[0, 0])
ax_g = fig.add_subplot(inner[1, 0])

ax_f = draw_1d_pdv(
    ax_f,
    model,
    X_train,
    feature='Operating_pressure',
    xlabel='Operating pressure (bar)',
    title='',
    color='tab:blue',
    xlim=(0, 40),
    x_ticks=np.arange(0, 41, 10)
)

ax_g = draw_1d_pdv(
    ax_g,
    model,
    X_train,
    feature='Stirring rate (rpm)',
    xlabel='Stirring rate (rpm)',
    title='',
    color='tab:olive',
    xlim=(200, 1000),
    x_ticks=np.arange(200, 1001, 200)
)

fig.subplots_adjust(
    left=0.065,
    right=0.985,
    bottom=0.045,
    top=0.985
)

fig.canvas.draw()
#Position setting
left_panel_x = ax1.get_position().x0
right_panel_x = ax_f.get_position().x0

panel_label_pad = 0.006

panel_label_positions = [
    ('a', left_panel_x, ax1.get_position().y1 + panel_label_pad),
    ('b', right_panel_x, ax2.get_position().y1 + panel_label_pad),
    ('c', left_panel_x, panel_c['marg_x'].get_position().y1 + panel_label_pad),
    ('d', right_panel_x, panel_d['marg_x'].get_position().y1 + panel_label_pad),
    ('e', left_panel_x, panel_e['marg_x'].get_position().y1 + panel_label_pad),
    ('f', right_panel_x, ax_f.get_position().y1 + panel_label_pad),
    ('g', right_panel_x, ax_g.get_position().y1 + panel_label_pad)]

for panel_label, panel_x, panel_y in panel_label_positions:
    fig.text(panel_x, panel_y, panel_label, fontsize=FS_PANEL, fontweight='bold', ha='left', va='bottom')
    
# plt.savefig('./figure_2.png', dpi=600, bbox_inches='tight')
# plt.savefig('./figure_2.pdf', dpi=600, bbox_inches='tight')
save_figure(fig, "figure_02")  # noqa: F405
