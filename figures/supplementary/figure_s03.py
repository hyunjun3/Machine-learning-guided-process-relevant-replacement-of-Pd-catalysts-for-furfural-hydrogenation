"""Generate Supplementary Figure 3."""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from figures.common import *  # noqa: F403

globals().update(load_ml_context())  # noqa: F405


#%% Supplementary figures - Fig. 3 (Operating time stratified SHAP summary plot)

model = XGBRegressor(random_state=SEED, n_jobs=-1)
model.load_model(f'./hyperparameter_tuning/output/xgb_model_seed_{SEED}.json')

MM_TO_INCH = 1 / 25.4

FIG_WIDTH = 180 * MM_TO_INCH
FIG_HEIGHT = 165 * MM_TO_INCH

FS_PANEL = 8.0
FS_TITLE = 7.0
FS_LABEL = 6.5
FS_TICK = 6.0
FS_FEATURE = 5.2
FS_CBAR = 6.0
FS_SAMPLE = 5.2

mpl.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': 'Arial',

    'font.size': FS_TICK,
    'axes.labelsize': FS_LABEL,
    'axes.titlesize': FS_TITLE,
    'axes.linewidth': 0.6,

    'xtick.labelsize': FS_TICK,
    'ytick.labelsize': FS_FEATURE,

    'xtick.major.size': 2.5,
    'ytick.major.size': 0,

    'xtick.major.width': 0.6,
    'ytick.major.width': 0.6,

    'xtick.direction': 'out',
    'ytick.direction': 'out',

    'pdf.fonttype': 42,
    'ps.fonttype': 42,
    'svg.fonttype': 'none',

    'savefig.facecolor': 'white'
})

rename_dict = {'Operating_temp': 'Operating temperature (°C)',
               'Operating_time': 'Operating time (h)',
               'Operating_pressure': 'Operating pressure (bar)',
               'Furfural (mg)': 'Furfural amount (mg)',
               'Active metal_Ni': 'Ni',
               'Substrate concentration (mg/ml)': 'Substrate conc. (mg/ml)',
               'Substrate to metal ratio (mmol/mmol)': 'Substrate-to-metal ratio',
               'Stirring rate (rpm)': 'Stirring rate (rpm)',
               'Reduction_temp': 'Reduction temperature (°C)',
               'Reduction_time': 'Reduction time (h)',
               'Calcination_temp': 'Calcination temperature (°C)',
               'Calcination_time': 'Calcination time (h)',
               'ethanol': 'Ethanol',
               'water': 'Water',
               '2-propanol': '2-propanol'}

time_conditions = [('Operating time < 1 h', X_train['Operating_time'] < 1),
                   ('Operating time 1–3 h', ((X_train['Operating_time'] >= 1) & (X_train['Operating_time'] < 3))),
                   ('Operating time 3–6 h', ((X_train['Operating_time'] >= 3) & (X_train['Operating_time'] <= 6))),
                   ('Operating time > 6 h', X_train['Operating_time'] > 6)]

#SHAP summary plot
explainer = shap.TreeExplainer(model)
shap_results = []

for title, condition in time_conditions:

    subset_model = X_train.loc[condition].copy()
    subset_display = subset_model.rename(
        columns=rename_dict
    )

    if len(subset_model) == 0:
        explanation = None

    else:
        sv = np.asarray(explainer.shap_values(subset_model))
        explanation = shap.Explanation(values=sv, data=subset_display.to_numpy(), feature_names=subset_display.columns.tolist())
        
    shap_results.append({'title': title,
                         'explanation': explanation,
                         'n_samples': len(subset_model)})

valid_values = [
    result['explanation'].values
    for result in shap_results
    if result['explanation'] is not None
]

if valid_values:

    global_max_abs = max(
        np.nanmax(np.abs(values))
        for values in valid_values
    )
    shap_limit = max(20, int(np.ceil(global_max_abs / 20) * 20))

else:
    shap_limit = 60


shap_ticks = np.arange(-shap_limit, shap_limit + 1, 20)

fig, axes = plt.subplots(
    nrows=2,
    ncols=2,
    figsize=(FIG_WIDTH, FIG_HEIGHT),
    dpi=300
)

axes = axes.ravel()

fig.subplots_adjust(left=0.205, right=0.895, bottom=0.085, top=0.955, wspace=0.55, hspace=0.30)

for idx, (ax, result) in enumerate(zip(axes, shap_results)):

    panel_label = string.ascii_lowercase[idx]
    title = result['title']
    explanation = result['explanation']
    n_samples = result['n_samples']

    if explanation is None:

        ax.text(
            0.5,
            0.5,
            'No samples',
            transform=ax.transAxes,
            ha='center',
            va='center',
            fontsize=FS_LABEL
        )

    else:
        shap.plots.beeswarm(
            explanation,
            max_display=15,
            ax=ax,
            show=False,
            color_bar=False,
            color=shap.plots.colors.red_blue,
            s=9,
            plot_size=None,
            group_remaining_features=False)   
        
        for line in list(ax.lines):
            xdata = np.asarray(line.get_xdata(), dtype=float)
            ydata = np.asarray(line.get_ydata(), dtype=float)

            is_horizontal = (
                ydata.size >= 2
                and np.allclose(ydata, ydata[0])
                and not np.allclose(xdata, xdata[0])
            )
        
            if is_horizontal:
                line.remove()

    ax.set_xlim(
        -shap_limit,
        shap_limit
    )

    ax.set_xticks(shap_ticks)

    ax.set_xlabel(
        'SHAP value',
        fontsize=FS_LABEL,
        labelpad=2
    )

    ax.set_ylabel('')

    ax.tick_params(
        axis='x',
        which='major',
        labelsize=FS_TICK,
        direction='out',
        width=0.6,
        length=2.5,
        pad=2
    )

    ax.tick_params(
        axis='y',
        which='major',
        labelsize=FS_FEATURE,
        length=0,
        pad=2
    )

    for tick_label in ax.get_yticklabels():
        tick_label.set_fontsize(FS_FEATURE)
        tick_label.set_fontweight('normal')

    ax.set_title(title, loc='center', fontsize=FS_TITLE, fontweight='bold', pad=3)
    ax.text(
        -0.055,
        1.025,
        panel_label,
        transform=ax.transAxes,
        fontsize=FS_PANEL,
        fontweight='bold',
        ha='left',
        va='bottom',
        clip_on=False
    )

    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(0.6)

shap_cmap = shap.plots.colors.red_blue
scalar_mappable = ScalarMappable(norm=Normalize(vmin=0, vmax=1), cmap=shap_cmap)
scalar_mappable.set_array([])

cbar_ax = fig.add_axes([
    0.925,   # left
    0.235,   # bottom
    0.014,   # width
    0.53     # height
])

cbar = fig.colorbar(
    scalar_mappable,
    cax=cbar_ax
)

cbar.set_ticks([0, 1])
cbar.set_ticklabels([
    'Low',
    'High'
])

cbar.set_label(
    'Feature value',
    fontsize=FS_CBAR,
    labelpad=3
)

cbar.ax.tick_params(
    labelsize=FS_FEATURE,
    direction='out',
    width=0.6,
    length=2
)

cbar.outline.set_linewidth(0.6)
# plt.savefig('./figure_supp_3.png', dpi=600, bbox_inches='tight')
# plt.savefig('./figure_supp_3.pdf', dpi=600, bbox_inches='tight')
save_figure(fig, "figure_s03")  # noqa: F405
