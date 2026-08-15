"""Generate Supplementary Figure 7."""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from figures.common import *  # noqa: F403

globals().update(load_ml_context())  # noqa: F405


#%% Supplementary figures - Fig. 7 (Potential space for 1X-4Ni candidates (Calculation))

with open('./dataset/molar_mass.pickle', 'rb') as f:
    molar_mass = pickle.load(f)
# ['Ca','Co','Cu','Fe','In','Ir','Ni', 'Pd','Pt', 'Re', 'Rh', 'Ru', 'Sn', 'Zn']

precursor_map = {
    'Ca': 'Ca(NO3)2',   
    'Co': 'Co(NO3)2',
    'Cu': 'Cu(NO3)2',
    'Fe': 'Fe(NO3)3', 
    'In': 'In(SO3CF3)3', 
    'Ir': 'IrCl3',
    'Ni': 'Ni(NO3)2', 
    'Pd': 'PdCl2', 
    'Pt': 'H2PtCl6', 
    'Re': 'NH4ReO4', 
    'Rh': 'RhCl3',   
    'Ru': 'RuCl3', 
    'Sn': 'SnCl4',    
    'Zn': 'Zn(NO3)2'} 

selected_support = 'Al2O3' 
selected_solvent = '2-propanol' 
solvent_amount   = 40
selected_preparation = 'wet impregnation'
calc_temp, calc_time = 500, 6
reduc_temp, reduc_time = 400, 4
oper_time =  6
stir_rate = 700
cat_amount, FF_amount = 100, 300


temp_range = np.arange(120, 201, 1)
pres_range = np.arange(10, 40.5, 0.5)

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

non_noble_target = ['Co', 'Fe', 'Cu', 'Ca', 'Zn', 'Re']
total_heatmap_data ={}

for asd in non_noble_target:
    ni_rows_list = []
    for t in temp_range:
        for p in pres_range:
            
            new_row = {col: 0 for col in dataset.columns[:-1]}
            
            AM_1, AM_2 = 'Ni', asd
            precursor_1 = precursor_map[AM_1]
            precursor_2 = precursor_map[AM_2]
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
            AM1_percent = 4.0 * 0.01    
            AM2_percent = 1.0 * 0.01 
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
    
    Ni_X_baseline_df['delta_y'] = (
        Ni_X_baseline_df['THFA_yield (%)'].values
        - Pd_baseline_df['THFA_yield (%)'].values
    )
    
    heatmap_data = Ni_X_baseline_df.pivot_table(
    index='Operating_pressure',
    columns='Operating_temp',
    values='delta_y')
    
    total_heatmap_data[f'1{asd}-4Ni'] = heatmap_data

non_noble_target = ['Co', 'Fe', 'Cu', 'Ca', 'Zn', 'Re']

vmin, vmax = -70, 20
v_range = vmax - vmin
clevels = np.linspace(vmin, vmax, 200)


color_nodes = [
    (0.0, '#001529'),                        
    (( -35 - vmin) / v_range, '#1e466e'),      
    (( -10 - vmin) / v_range, '#85a5c2'),     
    ((  0 - vmin) / v_range, '#ffffff'),     
    ((  5 - vmin) / v_range, '#e6c875'),      
    (( 12 - vmin) / v_range, '#d76445'),    
    (1.0, '#9c1515')                        
]
custom_cmap = mcolors.LinearSegmentedColormap.from_list('potential_map', color_nodes)

#%% Supplementary figures - Fig. 7 (Potential space for 1X-4Ni candidates (figure))
#Plot
MM_TO_INCH = 1 / 25.4

FIG_WIDTH = 180 * MM_TO_INCH
FIG_HEIGHT = 170 * MM_TO_INCH

FS_PANEL = 8.0
FS_TITLE = 6.5
FS_AXIS = 6.5
FS_TICK = 5.5
FS_CBAR = 6.5
FS_CBAR_TICK = 5.5

mpl.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': 'Arial',

    'font.size': FS_TICK,
    'axes.labelsize': FS_AXIS,
    'axes.titlesize': FS_TITLE,
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

panel_labels = string.ascii_lowercase[:len(non_noble_target)]

x_ticks = [120, 140, 160, 180, 200]
y_ticks = [10, 15, 20, 25, 30, 35, 40]

fig = plt.figure(figsize=(FIG_WIDTH, FIG_HEIGHT), dpi=300)
gs = fig.add_gridspec(nrows=3, ncols=2, width_ratios=[1.0, 1.0],
                      left=0.095, right=0.895, bottom=0.075, top=0.965, wspace=0.22, hspace=0.28)

axes = [
    fig.add_subplot(gs[0, 0]),
    fig.add_subplot(gs[0, 1]),
    fig.add_subplot(gs[1, 0]),
    fig.add_subplot(gs[1, 1]),
    fig.add_subplot(gs[2, 0]),
    fig.add_subplot(gs[2, 1])
]

contour = None

for idx, metal in enumerate(non_noble_target):

    ax = axes[idx]

    heatmap_key = f'1{metal}-4Ni'
    heatmap_data = total_heatmap_data[heatmap_key]

    temperature = heatmap_data.columns.to_numpy(dtype=float)
    pressure = heatmap_data.index.to_numpy(dtype=float)
    ZZ = heatmap_data.to_numpy(dtype=float)
    TT, PP = np.meshgrid(temperature, pressure)
    contour = ax.contourf(TT, PP, ZZ, levels=clevels,
                          cmap=custom_cmap, vmin=vmin, vmax=vmax, extend='both', antialiased=False)

    if hasattr(contour, 'collections'):
        for collection in contour.collections:
            collection.set_edgecolor('face')
            collection.set_linewidth(0.0)
            collection.set_antialiased(False)
            collection.set_rasterized(True)

    finite_values = ZZ[np.isfinite(ZZ)]

    if (finite_values.size > 0 and np.nanmin(finite_values) < 0 and np.nanmax(finite_values) > 0):
        zero_contour = ax.contour(TT, PP, ZZ, levels=[0], colors='black',
                                  linestyles='--', linewidths=0.8, zorder=4)

        if hasattr(zero_contour, 'collections'):
            for collection in zero_contour.collections:
                collection.set_rasterized(False)

    ax.text(-0.025, 1.035, panel_labels[idx], transform=ax.transAxes,
            fontsize=FS_PANEL, fontweight='bold', ha='left', va='bottom', clip_on=False)

    ax.set_xticks(x_ticks)
    ax.set_yticks(y_ticks)

    ax.tick_params(axis='both', which='major', labelsize=FS_TICK, direction='out',
                   width=0.6, length=2.5, pad=2)
    ax.tick_params(axis='x', labelbottom=True)
    ax.tick_params(axis='y', labelleft=True)

    if idx in [4, 5]:
        ax.set_xlabel('Operating temperature (°C)', fontsize=FS_AXIS, labelpad=3)
    else:
        ax.set_xlabel('')

    if idx in [0, 2, 4]:
        ax.set_ylabel('Operating pressure (bar)', fontsize=FS_AXIS, labelpad=3)
    else:
        ax.set_ylabel('')

    ax.grid(False)

    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(0.6)

for idx in range(len(non_noble_target), len(axes)):
    axes[idx].set_visible(False)

fig.canvas.draw()
right_axes = [ax for idx, ax in enumerate(axes) if (idx % 2 == 1 and ax.get_visible())]
right_edge = max(ax.get_position().x1 for ax in right_axes)
top_edge = max(ax.get_position().y1 for ax in axes if ax.get_visible())
bottom_edge = min(ax.get_position().y0 for ax in axes if ax.get_visible())

cbar_gap = 0.030
cbar_width = 0.025

cax = fig.add_axes([right_edge + cbar_gap, bottom_edge, cbar_width, top_edge - bottom_edge])
cbar = fig.colorbar(contour, cax=cax, extend='both')
cbar.set_ticks([-60, -40, -20, 0, 20])
cbar.set_label(r'Yield difference, $\Delta Y$ (%)', fontsize=FS_CBAR, labelpad=4)
cbar.ax.tick_params(axis='y', labelsize=FS_CBAR_TICK, direction='out', width=0.6, length=2.5, pad=2)
cbar.outline.set_linewidth(0.6)
# plt.savefig('./figure_supp_7.png', dpi=600, bbox_inches='tight')
# plt.savefig('./figure_supp_7.pdf', dpi=600, bbox_inches='tight')
save_figure(fig, "figure_s07")  # noqa: F405
