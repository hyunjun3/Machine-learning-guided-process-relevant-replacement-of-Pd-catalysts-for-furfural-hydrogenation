"""Generate Supplementary Figure 4."""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from figures.common import *  # noqa: F403

globals().update(load_ml_context())  # noqa: F405


#%% Supplementary figures - Fig. 4 (Absolute SHAP value / category-level SHAP value)

MM_TO_INCH = 1 / 25.4

# Double-column figure
FIG_WIDTH = 180 * MM_TO_INCH
FIG_HEIGHT = 95 * MM_TO_INCH

# Nature figure typography
FS_PANEL = 8.0       # panel labels: a, b
FS_TITLE = 7.0
FS_AXIS = 6.5
FS_TICK = 5.5
FS_FEATURE = 5.5
FS_VALUE = 5.5

mpl.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': 'Arial',

    'font.size': FS_TICK,
    'axes.labelsize': FS_AXIS,
    'axes.titlesize': FS_TITLE,
    'axes.linewidth': 0.6,

    'xtick.labelsize': FS_TICK,
    'ytick.labelsize': FS_FEATURE,

    'xtick.major.size': 2.5,
    'ytick.major.size': 2.5,
    'xtick.major.width': 0.6,
    'ytick.major.width': 0.6,
    'xtick.direction': 'out',
    'ytick.direction': 'out',

    # Editable text in vector figures
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

#SHAP summary plot
explainer = shap.TreeExplainer(model)
shap_values = np.asarray(explainer.shap_values(X_train))

mean_abs_shap = np.abs(shap_values).mean(axis=0)
shap_df = pd.DataFrame({
    'feature_original': X_train.columns,
    'feature': [
        rename_dict.get(name, name)
        for name in X_train.columns
    ],
    'mean_abs_shap': mean_abs_shap
}).sort_values(
    'mean_abs_shap',
    ascending=False
)

active_metal_set = set(active_metal)
support_set = set(cat_support)
precursor_set = set(precursor)
preparation_set = set(preparation)
solvent_set = set(solvent)

preparation_condition = {'Calcination_temp', 'Calcination_time', 'Reduction_temp', 'Reduction_time'} | preparation_set
reaction_condition = {'Furfural (mg)', 'Catalyst amount (mg)', 'Operating_temp', 'Operating_pressure',
                      'Operating_time', 'Stirring rate (rpm)', 'Substrate to metal ratio (mmol/mmol)',
                      'Substrate concentration (mg/ml)'} | solvent_set

def categorize_feature(name):

    if name in active_metal_set:
        return 'Active metal'

    if name in support_set:
        return 'Catalyst support'

    if name in precursor_set:
        return 'Metal precursor'

    if name in preparation_condition:
        return 'Preparation method'

    if name in reaction_condition:
        return 'Reaction conditions'

    return 'Other'


shap_df['Category'] = (shap_df['feature_original'].apply(categorize_feature))

category_sums = (shap_df.groupby('Category', observed=True)['mean_abs_shap'].sum())
category_percentages = (category_sums / category_sums.sum() * 100)
category_percentages = (category_percentages.sort_values(ascending=True))

fig = plt.figure(figsize=(FIG_WIDTH, FIG_HEIGHT), dpi=300)
gs = fig.add_gridspec(nrows=1, ncols=2, width_ratios=[1.20, 1.00], wspace=0.40)
ax1 = fig.add_subplot(gs[0, 0])
ax2 = fig.add_subplot(gs[0, 1])

#mean absolute SHAP value
top_n = 15
top_shap = (shap_df.head(top_n).sort_values('mean_abs_shap', ascending=True))

ax1.barh(
    y=top_shap['feature'],
    width=top_shap['mean_abs_shap'],
    height=0.72,
    color='#4C78A8',
    edgecolor='black',
    linewidth=0.30,
    zorder=2
)

ax1.set_xlabel('Mean absolute SHAP value', fontsize=FS_AXIS, labelpad=3)
ax1.set_ylabel('')

ax1.tick_params(axis='x', labelsize=FS_TICK, direction='out', width=0.6, length=2.5, pad=2)
ax1.tick_params(axis='y', labelsize=FS_FEATURE, direction='out', width=0.6, length=0, pad=2)
ax1.set_xlim(0,top_shap['mean_abs_shap'].max() * 1.06)
ax1.grid(False)


#category level plot
category_colors = {
    'Reaction conditions': '#0072B2',
    'Active metal': '#E69F00',
    'Catalyst support': '#009E73',
    'Preparation method': '#CC79A7',
    'Metal precursor': '#D55E00',
}

bar_colors = [category_colors.get(category, '#999999') for category in category_percentages.index]
bars = ax2.barh(y=category_percentages.index, width=category_percentages.values, height=0.62,
                color=bar_colors, edgecolor='black', linewidth=0.30, zorder=2)
ax2.set_xlabel('Category-level SHAP contribution', fontsize=FS_AXIS, labelpad=3)
ax2.set_ylabel('')
ax2.tick_params(axis='x', labelsize=FS_TICK, direction='out',
                width=0.6, length=2.5, pad=2)
ax2.tick_params(axis='y', labelsize=FS_FEATURE, direction='out',
                width=0.6, length=0, pad=2)

category_max = category_percentages.max()
ax2.set_xlim(0, np.ceil((category_max + 8) / 10) * 10)

for bar, percentage in zip(bars, category_percentages.values):
    ax2.text(percentage + 0.8, bar.get_y() + bar.get_height() / 2,
             f'{percentage:.1f}%', ha='left', va='center',
             fontsize=FS_VALUE, color='black')
ax2.grid(False)

for ax in [ax1, ax2]:
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(0.6)

    ax.set_axisbelow(True)

ax1.text(-0.03, 1.035, 'a', transform=ax1.transAxes, fontsize=FS_PANEL,
         fontweight='bold', fontstyle='normal', ha='left', va='bottom',
         color='black', clip_on=False)
ax2.text(-0.03, 1.035, 'b', transform=ax2.transAxes, fontsize=FS_PANEL,
         fontweight='bold', fontstyle='normal', ha='left', va='bottom',
         color='black', clip_on=False)

fig.subplots_adjust(left=0.190, right=0.980, bottom=0.155, top=0.930, wspace=0.58)

# plt.savefig('./figure_supp_4.png', dpi=600, bbox_inches='tight')
# plt.savefig('./figure_supp_4.pdf', dpi=600, bbox_inches='tight')
save_figure(fig, "figure_s04")  # noqa: F405
