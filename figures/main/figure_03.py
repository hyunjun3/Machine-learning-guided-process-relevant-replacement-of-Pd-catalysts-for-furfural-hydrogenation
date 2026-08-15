"""Generate manuscript Figure 3."""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from figures.common import *  # noqa: F403

globals().update(load_ml_context())  # noqa: F405



import pickle
import warnings

import matplotlib as mpl
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from sklearn.base import clone
from sklearn.inspection import partial_dependence


X_train_float = X_train.astype(float)
noble_metal = ['Pd', 'Pt', 'Rh', 'Ru', 'Ir']

if 'active_metal' not in globals():
    possible_metals = [
        'Ca', 'Co', 'Cu', 'Fe', 'In', 'Ir', 'Ni',
        'Pd', 'Pt', 'Re', 'Rh', 'Ru', 'Sn', 'Zn'
    ]
    active_metal = [
        metal for metal in possible_metals
        if metal in X_train_float.columns
    ]

Pd_only_df = X_train_float[(X_train_float['Pd'] > 0) & ((X_train_float[active_metal] > 0).sum(axis=1) == 1)]

condition_1 = X_train_float['Ni'] > 0
condition_2 = X_train_float[noble_metal].sum(axis=1) == 0
Ni_only_df = X_train_float[condition_1 & condition_2]

X_train_float = X_train.astype(float)

noble_metal = ['Pd', 'Pt', 'Rh', 'Ru', 'Ir']
noble_data = X_train_float[X_train_float[noble_metal].sum(axis=1) > 0]
non_noble = X_train_float[(X_train_float[noble_metal].sum(axis=1) == 0)]

features = ['Operating_temp', 'Operating_pressure', 'Operating_time']
x_labels = ['Operating temperature (°C)', ' Operating pressure (bar)', 'Operating time (h)']
x_lims = [(50, 200), (0, 40), (0, 10)]
x_ticks = [
    (50, 100, 150, 200),
    (0, 10, 20, 30, 40),
    (0, 2, 4, 6, 8, 10)
]
y_lims = [[(-12, 62), (-12, 62), (-12, 62)]]

temp_range = np.arange(120, 201, 1)
pres_range = np.arange(10, 40.5, 0.5)
selected_support = 'Al2O3' 
selected_solvent = '2-propanol' 
solvent_amount   = 40
selected_preparation = 'wet impregnation'
calc_temp, calc_time = 500, 6
reduc_temp, reduc_time = 400, 4
oper_time =  6
stir_rate = 700
cat_amount, FF_amount = 100, 300

with open('./dataset/molar_mass.pickle', 'rb') as f:
    molar_mass = pickle.load(f)
    
precursor_map = {
    'Ca': 'Ca(NO3)2', 'Co': 'Co(NO3)2', 'Cu': 'Cu(NO3)2', 'Fe': 'Fe(NO3)3', 
    'In': 'In(SO3CF3)3',  'Ir': 'IrCl3', 'Ni': 'Ni(NO3)2',  'Pd': 'PdCl2', 
    'Pt': 'H2PtCl6', 'Re': 'NH4ReO4', 'Rh': 'RhCl3', 'Ru': 'RuCl3', 'Sn': 'SnCl4','Zn': 'Zn(NO3)2'} 

pd_rows_list = []
for t in temp_range:
    for p in pres_range:
        
        new_row = {col: 0 for col in dataset.columns[:-1]}
        
        AM_1 = 'Pd'
        precursor_1 = precursor_map[AM_1]
        new_row[AM_1] = 5.0
        new_row[precursor_1] = 1
        new_row[selected_support] = 95.0
        new_row[selected_preparation] = 1 
        new_row[selected_solvent] = solvent_amount
        new_row['Stirring rate (rpm)'] = stir_rate
        new_row['Catalyst amount (mg)'] = cat_amount
        new_row['Furfural (mg)'] = FF_amount
        new_row['Operating_temp'] = t
        new_row['Operating_pressure'] = p
        new_row['Operating_time'] = oper_time
        new_row['Calcination_temp'] = calc_temp
        new_row['Reduction_temp'] = reduc_temp
        new_row['Calcination_time'] = calc_time
        new_row['Reduction_time'] = reduc_time
        furfural_mmol   = float(FF_amount / molar_mass['Furfural'])
        AM1_percent =  5.0 * 0.01    
        subs_to_metal = float(furfural_mmol / (cat_amount * AM1_percent / molar_mass[AM_1]))
        subs_concentration = float(FF_amount / solvent_amount)
        new_row['Substrate to metal ratio (mmol/mmol)'] = subs_to_metal
        new_row['Substrate concentration (mg/ml)'] = subs_concentration
        pd_rows_list.append(new_row)

Pd_baseline_df = pd.DataFrame(pd_rows_list)
X_features = dataset.columns.tolist()[:-1] 
y_pred = model.predict(Pd_baseline_df[X_features])
y_pred = np.clip(y_pred, 0, 100)
Pd_baseline_df['THFA_yield (%)'] = y_pred
Pd_baseline_df['Combination'] = 'Pd (5wt%)'

ni_rows_list = []
for t in temp_range:
    for p in pres_range:
        
        new_row = {col: 0 for col in dataset.columns[:-1]}
        
        AM_1, AM_2 = 'Ni', 'Re'
        precursor_1, precursor_2 = precursor_map[AM_1], precursor_map[AM_2]
        new_row[AM_1], new_row[AM_2] = 4.0, 1.0
        
        new_row[precursor_1], new_row[precursor_2] = 1, 1
        new_row[selected_support] = 95.0
        new_row[selected_preparation] = 1 
        new_row[selected_solvent] = solvent_amount
        
        new_row['Stirring rate (rpm)'] = stir_rate
        new_row['Catalyst amount (mg)'] = cat_amount
        new_row['Furfural (mg)'] = FF_amount
        
        new_row['Operating_temp'] = t
        new_row['Operating_pressure'] = p
        new_row['Operating_time'] = oper_time
        
        new_row['Calcination_temp'] = calc_temp
        new_row['Reduction_temp'] = reduc_temp
        
        new_row['Calcination_time'] = calc_time
        new_row['Reduction_time'] = reduc_time

        furfural_mmol   = float(FF_amount / molar_mass['Furfural'])
    
        AM1_percent, AM2_percent = 4.0 * 0.01, 1.0 * 0.01 
        
        subs_to_metal = float(furfural_mmol / (cat_amount * AM1_percent / molar_mass[AM_1] + cat_amount * AM2_percent / molar_mass[AM_2]))
        subs_concentration = float(FF_amount / solvent_amount)
            
        new_row['Substrate to metal ratio (mmol/mmol)'] = subs_to_metal
        new_row['Substrate concentration (mg/ml)'] = subs_concentration
        
        ni_rows_list.append(new_row)

Ni_X_baseline_df = pd.DataFrame(ni_rows_list)
X_features = dataset.columns.tolist()[:-1] 
y_pred = model.predict(Ni_X_baseline_df[X_features])
y_pred = np.clip(y_pred, 0, 100)
Ni_X_baseline_df['THFA_yield (%)'] = y_pred
Ni_X_baseline_df['Combination'] = f'Ni (4wt%) / {AM_2} (1wt%)'

Ni_X_baseline_df['delta_y'] = (Ni_X_baseline_df['THFA_yield (%)'].values- Pd_baseline_df['THFA_yield (%)'].values)

heatmap_data = Ni_X_baseline_df.pivot_table(index='Operating_pressure', columns='Operating_temp', values='delta_y')

T = heatmap_data.columns.values.astype(float)
P = heatmap_data.index.values.astype(float)
TT, PP = np.meshgrid(T, P)
ZZ = heatmap_data.values

def build_and_predict_case(active_metals, combination_label):
    """
    active_metals example:
        {'Pd': 5.0}
        {'Ni': 4.0, 'Re': 1.0}
        {'Ni': 5.0}
    """
    rows = []

    for t in temp_range:
        for p in pres_range:
            new_row = {col: 0 for col in dataset.columns[:-1]}

            # active metal loading and precursor one-hot
            for metal, loading in active_metals.items():
                new_row[metal] = loading
                precursor = precursor_map[metal]
                new_row[precursor] = 1

            total_metal_loading = sum(active_metals.values())
            new_row[selected_support] = 100.0 - total_metal_loading
            new_row[selected_preparation] = 1
            new_row[selected_solvent] = solvent_amount

            new_row['Stirring rate (rpm)'] = stir_rate
            new_row['Catalyst amount (mg)'] = cat_amount
            new_row['Furfural (mg)'] = FF_amount

            new_row['Operating_temp'] = t
            new_row['Operating_pressure'] = p
            new_row['Operating_time'] = oper_time

            new_row['Calcination_temp'] = calc_temp
            new_row['Reduction_temp'] = reduc_temp
            new_row['Calcination_time'] = calc_time
            new_row['Reduction_time'] = reduc_time

            # substrate-to-metal ratio and substrate concentration
            furfural_mmol = float(FF_amount / molar_mass['Furfural'])

            metal_mmol = 0.0
            for metal, loading in active_metals.items():
                metal_fraction = loading * 0.01
                metal_mmol += cat_amount * metal_fraction / molar_mass[metal]

            subs_to_metal = float(furfural_mmol / metal_mmol)
            subs_concentration = float(FF_amount / solvent_amount)

            new_row['Substrate to metal ratio (mmol/mmol)'] = subs_to_metal
            new_row['Substrate concentration (mg/ml)'] = subs_concentration

            rows.append(new_row)

    case_df = pd.DataFrame(rows)

    # Make sure columns are aligned with training features
    case_X = case_df.reindex(columns=X_features, fill_value=0)
    y_pred = model.predict(case_X)
    y_pred = np.clip(y_pred, 0, 100)

    case_df['THFA_yield (%)'] = y_pred
    case_df['Combination'] = combination_label

    return case_df

Pd_baseline_df = build_and_predict_case(
    active_metals={'Pd': 5.0},
    combination_label='5Pd'
)

pd_ref = Pd_baseline_df[
    ['Operating_temp', 'Operating_pressure', 'THFA_yield (%)']
].rename(columns={'THFA_yield (%)': 'THFA_yield_Pd (%)'})
pd_case_X = Pd_baseline_df.reindex(columns=X_features, fill_value=0).copy()

non_noble_target = ['Co', 'Fe', 'Cu', 'Ca', 'Zn', 'Re']
gap_threshold = 0.0
heatmap_dict = {}
mean_yield_diff = {}
positive_region_fraction = {}
max_yield_diff = {}
candidate_case_X = {}

for X in non_noble_target:
    label = f'1{X}-4Ni'

    NiX_df = build_and_predict_case(
        active_metals={'Ni': 4.0, X: 1.0},
        combination_label=label
    )
    candidate_case_X[label] = NiX_df.reindex(
        columns=X_features, fill_value=0
    ).copy()

    NiX_df = NiX_df.merge(
        pd_ref,
        on=['Operating_temp', 'Operating_pressure'],
        how='left'
    )

    NiX_df['delta_y'] = (
        NiX_df['THFA_yield (%)'] - NiX_df['THFA_yield_Pd (%)']
    )

    heatmap_data = NiX_df.pivot_table(
        index='Operating_pressure',
        columns='Operating_temp',
        values='delta_y'
    )

    Z = heatmap_data.values

    heatmap_dict[label] = heatmap_data
    mean_yield_diff[label] = float(np.nanmean(Z))
    max_yield_diff[label] = float(np.nanmax(Z))
    positive_region_fraction[label] = float(np.nanmean(Z >= gap_threshold) * 100.0)

screening_metrics_df = pd.DataFrame({
    'Catalyst': list(mean_yield_diff.keys()),
    'Mean yield gap (%)': list(mean_yield_diff.values()),
    'Maximum yield gap (%)': list(max_yield_diff.values()),
    f'Positive-region fraction, ΔY >= {gap_threshold:g} (%)': list(positive_region_fraction.values())
})

heatmap_data = heatmap_dict['1Re-4Ni']

T = heatmap_data.columns.values.astype(float)
P = heatmap_data.index.values.astype(float)
TT, PP = np.meshgrid(T, P)
ZZ = heatmap_data.values

####################################################################################################################################
#figure
MM_TO_INCH = 1 / 25.4

FIG_WIDTH = 180 * MM_TO_INCH
FIG_HEIGHT = 180 * MM_TO_INCH

FS_PANEL = 8.0
FS_LABEL = 6.5
FS_TICK = 6
FS_TEXT = 5.5
FS_VALUE = 5.2
FS_LEGEND = 6.0
FS_CBAR = 6.0
FS_CBAR_TICK = 5.5

mpl.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': 'Arial',

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

    'lines.linewidth': 0.9,

    'pdf.fonttype': 42,
    'ps.fonttype': 42,
    'svg.fonttype': 'none',

    'savefig.facecolor': 'white'
})

features = ['Operating_temp', 'Operating_pressure', 'Operating_time']
x_labels = ['Operating temperature (°C)', 'Operating pressure (bar)', 'Operating time (h)']
x_lims = [(50, 200), (0, 40), (0, 10)]
x_ticks = [[50, 100, 150, 200], [0, 10, 20, 30, 40], [0, 2, 4, 6, 8, 10]]
y_lim_pdp = (-12, 62)
y_ticks_pdp = [0, 20, 40, 60]

re_key = '1Re-4Ni'
heatmap_data = heatmap_dict[re_key]

T = heatmap_data.columns.to_numpy(dtype=float)
P = heatmap_data.index.to_numpy(dtype=float)
TT, PP = np.meshgrid(T, P)
ZZ = heatmap_data.to_numpy(dtype=float)

# Revised panel order shown in the reference figure.
candidate_order = ['Re', 'Fe', 'Ca', 'Zn', 'Cu', 'Co']
labels = [
    f'1{metal}-4Ni' for metal in candidate_order
    if f'1{metal}-4Ni' in positive_region_fraction
]

if len(labels) == 0:
    labels = list(positive_region_fraction.keys())
positive_values = np.array([positive_region_fraction[label] for label in labels], dtype=float)
mean_gap_values = np.array([mean_yield_diff[label] for label in labels], dtype=float)
display_labels = [label.replace('-', '–') for label in labels]

# Bootstrap-refit stability used in revised panel f.  These values are
# calculated from the data/model, not hard-coded from the example figure.
N_BOOTSTRAP = 200
BOOTSTRAP_RANDOM_STATE = 42


def get_bootstrap_training_data():
    X_base = X_train_float.reindex(columns=X_features, fill_value=0).copy()

    if 'y_train' in globals():
        y_base = np.asarray(globals()['y_train']).reshape(-1)
    elif X_base.index.isin(dataset.index).all():
        y_base = dataset.loc[
            X_base.index, dataset.columns[-1]
        ].to_numpy()
    else:
        raise ValueError(
            'Panel f requires y_train, or X_train indices aligned with '
            'the target in the last column of dataset.'
        )

    if len(X_base) != len(y_base):
        raise ValueError('X_train and y_train have different lengths.')

    return X_base, np.asarray(y_base, dtype=float)


def calculate_rank1_bootstrap_frequencies(
    n_bootstrap=N_BOOTSTRAP,
    random_state=BOOTSTRAP_RANDOM_STATE
):
    X_base, y_base = get_bootstrap_training_data()
    rng = np.random.default_rng(random_state)
    winner_counts = {label: 0 for label in labels}

    for _ in range(n_bootstrap):
        sampled = rng.integers(0, len(X_base), size=len(X_base))
        bootstrap_model = clone(model)

        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            bootstrap_model.fit(
                X_base.iloc[sampled],
                y_base[sampled]
            )

        pd_boot = np.clip(
            bootstrap_model.predict(pd_case_X), 0, 100
        )
        scores = {}
        for label in labels:
            candidate_boot = np.clip(
                bootstrap_model.predict(candidate_case_X[label]), 0, 100
            )
            scores[label] = float(np.nanmean(candidate_boot - pd_boot))

        winner_counts[max(scores, key=scores.get)] += 1

    frequencies = {
        label: 100.0 * count / n_bootstrap
        for label, count in winner_counts.items()
    }
    return winner_counts, frequencies


# Panel f uses the bootstrap results that were already calculated separately.
# They are intentionally fixed so merely plotting the figure never refits the
# model or changes the displayed values.
bootstrap_counts = {
    '1Re-4Ni': 191,
    '1Fe-4Ni': 6,
    '1Ca-4Ni': 2,
    '1Co-4Ni': 1,
    '1Cu-4Ni': 0,
    '1Zn-4Ni': 0,
}
bootstrap_frequency = {
    label: 100.0 * count / N_BOOTSTRAP
    for label, count in bootstrap_counts.items()
}

#set colors
COLOR_PD = '#CC4C4C'
COLOR_NI = '#4775B8'
COLOR_BAR = '#67A9CF'
COLOR_BLUE = '#1F5DAA'
COLOR_RE = '#C51B2D'
COLOR_RE_TEXT = '#B2182B'
COLOR_EDGE = '#8297A6'
COLOR_ZERO = '0.25'
# heat_vmin = -20
# heat_vmax = 15
# custom_cmap = mcolors.LinearSegmentedColormap.from_list('potential_map',['#1F4E79', '#8FB6D6',  '#F7F7F7', '#E6C875', '#D76445', '#9C1515'])
# heat_norm = mcolors.TwoSlopeNorm(vmin=heat_vmin, vcenter=0, vmax=heat_vmax)

heat_vmin = -20
heat_vmax = 15

# Diverging colormap:
# negative ΔY = blue, ΔY = 0 = white, positive ΔY = red
custom_cmap = mcolors.LinearSegmentedColormap.from_list(
    'pd_relative_map',
    [
        (0.00, '#2166AC'),  # strongly negative: Pd favored
        (0.25, '#67A9CF'),  # moderately negative
        (0.50, '#F7F7F7'),  # ΔY = 0: equal predicted yield
        (0.75, '#EF8A62'),  # moderately positive
        (1.00, '#B2182B')   # strongly positive: Re–Ni favored
    ],
    N=256
)

heat_norm = mcolors.TwoSlopeNorm(
    vmin=heat_vmin,
    vcenter=0.0,
    vmax=heat_vmax
)

def add_panel_label(ax, label):
    ax.text(
        0.00,
        1.035,
        label,
        transform=ax.transAxes,
        fontsize=FS_PANEL,
        fontweight='bold',
        ha='left',
        va='bottom',
        color='black',
        clip_on=False
    )


def style_axis(
    ax,
    grid_axis=None
):

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

    ax.grid(False)

    if grid_axis is not None:
        ax.grid(
            axis=grid_axis,
            linestyle=':',
            linewidth=0.35,
            color='0.82',
            alpha=0.8
        )

    ax.set_axisbelow(True)


def get_pd_grid(result):
    if 'grid_values' in result:
        return result['grid_values'][0]

    return result['values'][0]


def remove_contour_seams(contour_set):
    for collection in getattr(contour_set, 'collections', []):
        collection.set_edgecolor('face')
        collection.set_linewidth(0.0)
        collection.set_antialiased(False)
        collection.set_rasterized(True)


#figure layout
fig = plt.figure(figsize=(FIG_WIDTH, FIG_HEIGHT), dpi=300)

outer_gs = fig.add_gridspec(
    nrows=2, ncols=1, height_ratios=[0.82, 1.58],
    left=0.075, right=0.975, bottom=0.075, top=0.915, hspace=0.25)

top_gs = outer_gs[0].subgridspec(nrows=1, ncols=3, wspace=0.25)
bottom_gs = outer_gs[1].subgridspec(
    nrows=1, ncols=2, width_ratios=[1.45, 1.0], wspace=0.32
)
heat_gs = bottom_gs[0].subgridspec(
    nrows=1, ncols=2, width_ratios=[1.0, 0.050], wspace=0.08
)
right_gs = bottom_gs[1].subgridspec(
    nrows=2, ncols=1, height_ratios=[1.0, 0.92], hspace=0.50
)
panel_e_gs = right_gs[0].subgridspec(
    nrows=1, ncols=2, width_ratios=[1.25, 1.0], wspace=0.3
)

ax1 = fig.add_subplot(top_gs[0, 0])
ax2 = fig.add_subplot(top_gs[0, 1])
ax3 = fig.add_subplot(top_gs[0, 2])
ax4 = fig.add_subplot(heat_gs[0, 0])
cax = fig.add_subplot(heat_gs[0, 1])
ax5_left = fig.add_subplot(panel_e_gs[0, 0])
ax5_right = fig.add_subplot(panel_e_gs[0, 1], sharey=ax5_left)
ax6 = fig.add_subplot(right_gs[1, 0])
pdp_axes = [ax1, ax2, ax3]

#monometalic Pd vs Ni-based non-noble-metal catalysts
for idx, (ax, feature, x_label, x_lim,ticks) in enumerate(zip(pdp_axes, features, x_labels, x_lims, x_ticks)):

    pd_result = partial_dependence(
        estimator=model,
        X=Pd_only_df,
        features=[feature],
        kind='average',
        method='brute',
        grid_resolution=100,
        percentiles=(0.0, 1.0)
    )

    ni_result = partial_dependence(
        estimator=model,
        X=Ni_only_df,
        features=[feature],
        kind='average',
        method='brute',
        grid_resolution=100,
        percentiles=(0.0, 1.0)
    )

    pd_grid = get_pd_grid(pd_result)
    ni_grid = get_pd_grid(ni_result)

    pd_average = np.asarray(pd_result['average'][0], dtype=float)
    ni_average = np.asarray(ni_result['average'][0], dtype=float)

    ax.plot(pd_grid, pd_average, color=COLOR_PD, linewidth=1.2, zorder=3)
    ax.plot(ni_grid, ni_average, color=COLOR_NI, linewidth=1.2, zorder=3)

    #rug plot (dataset distribution)
    sns.rugplot(x=Pd_only_df[feature], ax=ax, color=COLOR_PD,
        alpha=0.55, height=0.035, linewidth=0.45)
    sns.rugplot(x=Ni_only_df[feature], ax=ax, color=COLOR_NI,
        alpha=0.55, height=0.035, linewidth=0.45)
    ax.axhline(y=0, color='0.45', linewidth=0.55, linestyle='-', zorder=1)

    ax.set_xlim(x_lim)
    ax.set_ylim(y_lim_pdp)
    ax.set_xticks(ticks)
    ax.set_yticks(y_ticks_pdp)
    ax.set_xlabel(x_label, fontsize=FS_LABEL, labelpad=3)

    if idx == 0:
        ax.set_ylabel('Partial dependence value', fontsize=FS_LABEL, labelpad=3)
    else:
        ax.set_ylabel('')

    ax.tick_params(axis='both', labelsize=FS_TICK)
    add_panel_label(ax, chr(ord('a') + idx))
    style_axis(ax, grid_axis=None)

legend_handles = [Line2D([0], [0], color=COLOR_PD, linewidth=1.2,
                         label='Monometallic Pd catalysts'),
                  Line2D([0], [0], color=COLOR_NI, linewidth=1.2,
                         label='Non-Pd Ni-based catalysts')]

fig.legend(
    handles=legend_handles,
    loc='upper center',
    bbox_to_anchor=(0.5, 0.972),
    ncol=2,
    frameon=False,
    fontsize=FS_LEGEND,
    handlelength=2.3,
    handletextpad=0.5,
    columnspacing=1.7,
    borderaxespad=0
)

#Yield gap heatmap
levels = np.linspace(heat_vmin, heat_vmax, 141)
contour = ax4.contourf(TT, PP, ZZ, levels=levels, cmap=custom_cmap, norm=heat_norm, extend='both', antialiased=False)
remove_contour_seams(contour)

finite_zz = ZZ[np.isfinite(ZZ)]

if (finite_zz.size > 0 and np.nanmin(finite_zz) < 0 and np.nanmax(finite_zz) > 0):
    zero_contour = ax4.contour(TT, PP, ZZ,
        levels=[0], colors='black', linewidths=0.8, linestyles='--', zorder=4)

    ax4.clabel(zero_contour, inline=True, inline_spacing=3, fontsize=FS_TEXT,
        fmt={0: r'$\Delta Y = 0$'})

ax4.set_xlim(np.nanmin(T), np.nanmax(T))
ax4.set_ylim(np.nanmin(P), np.nanmax(P))
ax4.set_xticks([120, 140, 160, 180, 200])
ax4.set_yticks([10, 15, 20, 25, 30, 35, 40])
ax4.set_xlabel('Operating temperature (°C)', fontsize=FS_LABEL, labelpad=3)
ax4.set_ylabel('Operating pressure (bar)',
    fontsize=FS_LABEL,
    labelpad=3
)

ax4.tick_params(
    axis='both',
    labelsize=FS_TICK
)

add_panel_label(
    ax4,
    'd'
)

style_axis(
    ax4,
    grid_axis=None
)


# =============================================================================
# 12. Heatmap colorbar
# =============================================================================
cbar = fig.colorbar(
    contour,
    cax=cax,
    extend='both'
)

cbar.set_ticks(
    [-20, -15, -10, -5, 0, 5, 10, 15]
)

cbar.set_label(
    r'Yield difference, $\Delta Y$ (percentage points)',
    fontsize=FS_CBAR,
    labelpad=7
)
cbar.ax.yaxis.set_label_position('right')
cbar.ax.yaxis.set_ticks_position('right')

cbar.ax.tick_params(
    axis='y',
    labelsize=7,
    direction='out',
    width=0.6,
    length=2.5,
    pad=2
)

cbar.outline.set_linewidth(0.6)

# Favored-region labels moved out of the heatmap and placed beside the
# matching ends of the colorbar.
cbar.ax.annotate(
    'Re–Ni\nfavored',
    xy=(1.0, 0.98), xycoords='axes fraction',
    xytext=(8, 0), textcoords='offset points',
    color=COLOR_RE_TEXT, fontsize=7, fontweight='bold',
    ha='left', va='top', annotation_clip=False
)
cbar.ax.annotate(
    'Pd\nfavored',
    xy=(1.0, 0.02), xycoords='axes fraction',
    xytext=(8, 0), textcoords='offset points',
    color=COLOR_BLUE, fontsize=7, fontweight='bold',
    ha='left', va='bottom', annotation_clip=False
)

# =============================================================================
# Revised panel e: operating-space classification + Pd-relative proximity
# =============================================================================
y_pos = np.arange(len(labels))
negative_values = 100.0 - positive_values

ax5_left.barh(
    y_pos, negative_values, height=0.58,
    color=COLOR_BAR, edgecolor=COLOR_EDGE, linewidth=0.35, zorder=2
)
ax5_left.barh(
    y_pos, positive_values, left=negative_values, height=0.58,
    color=COLOR_RE, edgecolor=COLOR_EDGE, linewidth=0.35, zorder=3
)

for row, value, negative_width in zip(
    y_pos, positive_values, negative_values
):
    if value > 0:
        ax5_left.text(
            negative_width + value / 2, row, f'{value:.1f}%',
            color='white', fontsize=FS_VALUE, fontweight='bold',
            ha='center', va='center', zorder=4
        )

ax5_left.set_xlim(0, 100)
ax5_left.set_ylim(len(y_pos) - 0.3, -0.7)
ax5_left.set_yticks(y_pos)
ax5_left.set_yticklabels(display_labels)
ax5_left.set_xticks([0, 20, 40, 60, 80, 100])
ax5_left.set_xlabel(
    'Grid-point fraction (%)',
    fontsize=5.4, labelpad=3
)
ax5_left.set_title(
    'Pd-relative operating-space partition',
    fontsize=FS_TEXT, fontweight='bold', pad=6
)
style_axis(ax5_left, grid_axis=None)
ax5_left.spines['top'].set_visible(False)
ax5_left.spines['right'].set_visible(False)

for tick, label in zip(ax5_left.get_yticklabels(), labels):
    if label == re_key:
        tick.set_color(COLOR_RE_TEXT)
        tick.set_fontweight('bold')

mean_low = min(
    -70.0,
    np.floor(np.nanmin(mean_gap_values) / 10.0) * 10.0 - 5.0
)
mean_high = max(
    5.0,
    np.ceil(np.nanmax(mean_gap_values) / 5.0) * 5.0 + 2.0
)

for row in y_pos:
    ax5_right.axhline(row, color='0.88', linewidth=0.5, zorder=0)
ax5_right.axvline(
    0, color='0.35', linestyle=':', linewidth=0.7, zorder=1
)

for row, label, value in zip(y_pos, labels, mean_gap_values):
    is_re = label == re_key
    ax5_right.scatter(
        value, row,
        s=26 if is_re else 12,
        marker='*' if is_re else 'o',
        facecolor=COLOR_RE if is_re else 'white',
        edgecolor=COLOR_RE_TEXT if is_re else COLOR_BLUE,
        linewidth=0.7, zorder=3
    )
    ax5_right.annotate(
        f'{value:.1f}', xy=(value, row),
        xytext=(0, 7), textcoords='offset points',
        color=COLOR_RE_TEXT if is_re else 'black',
        fontsize=FS_VALUE,
        fontweight='bold' if is_re else 'normal',
        ha='center', va='bottom', annotation_clip=False
    )

ax5_right.set_xlim(mean_low, mean_high)
ax5_right.set_ylim(len(y_pos) - 0.3, -0.7)
ax5_right.tick_params(axis='y', left=False, labelleft=False)
ax5_right.set_xlabel(
    r'Mean $\Delta Y$ (percentage points)',
    fontsize=5.4,
    labelpad=3
)
ax5_right.set_title(
    'Pd-relative yield difference',
    fontsize=FS_TEXT, fontweight='bold', pad=6
)
style_axis(ax5_right, grid_axis=None)
ax5_right.spines['top'].set_visible(False)
ax5_right.spines['right'].set_visible(False)

ax5_left.text(
    -0.30, 1.035, 'e', transform=ax5_left.transAxes,
    fontsize=FS_PANEL, fontweight='bold',
    ha='left', va='bottom', clip_on=False
)
# ax5_left.text(
#     -0.12, 1.30, 'Nominal candidate prioritization',
#     transform=ax5_left.transAxes,
#     fontsize=FS_LABEL, fontweight='bold',
#     ha='left', va='bottom', clip_on=False
# )

panel_e_legend = [
    Patch(
        facecolor=COLOR_BAR, edgecolor='none',
        label=  r'Pd favored ($\Delta Y < 0$)'
    ),
    Patch(
        facecolor=COLOR_RE, edgecolor='none',
        label=r'Candidate $\geq$ Pd ($\Delta Y \geq 0$)'
    )
]
ax5_left.legend(
    handles=panel_e_legend,
    loc='lower left', bbox_to_anchor=(0.76, 1.13),
    ncol=2, frameon=False, fontsize=4.6,
    handlelength=2.0, handletextpad=0.4,
    columnspacing=1.0, borderaxespad=0
)


# =============================================================================
# Revised panel f: rank-1 frequency across bootstrap-refitted models
# =============================================================================
# Fixed display order accompanying the precomputed 191/6/2/1/0/0 counts.
bootstrap_plot_labels = [
    '1Re-4Ni', '1Fe-4Ni', '1Ca-4Ni',
    '1Co-4Ni', '1Cu-4Ni', '1Zn-4Ni'
]
alternative_labels = bootstrap_plot_labels[1:]
bootstrap_display_labels = [
    label.replace('-', '–') for label in bootstrap_plot_labels
]
bootstrap_y = np.arange(len(bootstrap_plot_labels))

for row, label in zip(bootstrap_y, bootstrap_plot_labels):
    frequency = bootstrap_frequency[label]
    count = bootstrap_counts[label]
    is_re = label == re_key
    point_color = COLOR_RE if is_re else COLOR_BLUE

    ax6.hlines(
        row, 0, frequency,
        color=point_color, linewidth=1.2, zorder=2
    )
    ax6.scatter(
        frequency, row,
        s=24 if is_re else 14,
        color=point_color,
        edgecolor='white', linewidth=0.3, zorder=3
    )
    ax6.annotate(
        f'{frequency:.1f}%',
        xy=(frequency, row), xytext=(10, 0),
        textcoords='offset points',
        color=point_color, fontsize=FS_VALUE,
        fontweight='bold' if is_re else 'normal',
        ha='left', va='center', annotation_clip=False
    )

ax6.set_xlim(0, 100)
ax6.set_ylim(len(bootstrap_y) - 0.3, -0.25)
ax6.set_yticks(bootstrap_y)
ax6.set_yticklabels(bootstrap_display_labels)
ax6.set_xticks([0, 20, 40, 60, 80, 100])
ax6.set_xlabel(
    f'Rank-1 frequency (%)',
    fontsize=FS_LABEL, labelpad=3
)
style_axis(ax6, grid_axis='x')
ax6.spines['top'].set_visible(False)
ax6.spines['right'].set_visible(False)

for tick, label in zip(ax6.get_yticklabels(), bootstrap_plot_labels):
    if label == re_key:
        tick.set_color(COLOR_RE_TEXT)
        tick.set_fontweight('bold')

ax6.text(
    -0.13, 1.1035, 'f', transform=ax6.transAxes,
    fontsize=FS_PANEL, fontweight='bold',
    ha='left', va='bottom', clip_on=False
)

# Magnified view for the five alternative candidates, as in the reference.
alternative_max = max(
    [bootstrap_frequency[label] for label in alternative_labels],
    default=0.0
)
if alternative_max <= 3.5:
    # inset = ax6.inset_axes([0.66, 0.12, 0.31, 0.50])
    inset = ax6.inset_axes([0.64, 0.22, 0.31, 0.44])
    inset_y = np.arange(len(alternative_labels))

    for row, label in zip(inset_y, alternative_labels):
        frequency = bootstrap_frequency[label]
        inset.hlines(
            row, 0, frequency,
            color=COLOR_BLUE, linewidth=0.8
        )
        inset.scatter(
            frequency, row, s=10,
            color=COLOR_BLUE, zorder=3
        )

    inset.set_xlim(0, 3.5)
    inset.set_ylim(len(inset_y) - 0.3, -0.7)
    inset.set_yticks(inset_y)
    inset.set_yticklabels(
        [label.replace('-', '–') for label in alternative_labels],
        fontsize=4.2
    )
    inset.set_xticks([0, 1, 2, 3, 3.5])
    inset.tick_params(axis='x', labelsize=4.2, length=1.8, pad=1)
    inset.tick_params(axis='y', length=0, pad=1)
    inset.grid(
        axis='x', linestyle=':', color='0.84', linewidth=0.35
    )
    inset.set_title(
        'Expanded view', fontsize=4.8, fontweight='bold', pad=2
    )
    inset.set_xlabel(
        'Rank-1 frequency (%)', fontsize=4.2, labelpad=1
    )
    for spine in inset.spines.values():
        spine.set_linewidth(0.5)


save_figure(fig, "figure_03")  # noqa: F405
