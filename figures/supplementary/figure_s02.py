"""Generate Supplementary Figure 2."""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from figures.common import *  # noqa: F403

globals().update(load_ml_context())  # noqa: F405


#%% Supplementary figures - Fig. 2 (All parity plots (Load model result))

model_name = ['XGBoost (XGB)', 'CatBoost (CB)', 'Random forest (RF)', 
              'LightGBM (LGBM)', 'Decision tree (DT)', 'Linear regression (LR)',
              'Lasso regression (Lasso)', 'Support vector regression (SVR)', 'Ridge regression (Ridge)']

base_path = './hyperparameter_tuning/output/'
output_files = ['xgb_model_seed_23.json', 'catboost_model_seed_23.cbm', 'RF_model_seed23.pkl', 
                'lightGBM_model_seed23.txt', 'DT_model_seed23.pkl', 'lr_model_seed_23.pkl', 
                'lasso_model_seed_23.pkl', 'svr_model_seed_23.pkl', 'ridge_model_seed_23.pkl']

# Prediction results
pred_results = []

for name, file in zip(model_name, output_files):
    model_path = base_path + file
    X_test  = pd.read_csv('./dataset/ML_dataset_final_x_test.csv')
    
    if 'cbm' in file:
        model = CatBoostRegressor()
        model.load_model(model_path)

    if 'xgb' in file:
        model = XGBRegressor()
        model.load_model(model_path)
    
    if 'lightGBM' in file:
        model = lgb.Booster(model_file=f'./hyperparameter_tuning/output/lightGBM_model_seed{SEED}.txt')
    
    if 'DT' in file or 'RF' in file:
        with open(model_path, "rb") as f:
            model = pickle.load(f)
    
    if 'lr' in file:
        with open(f'./hyperparameter_tuning/output/lr_model_seed_{SEED}.pkl',  'rb') as f:
            model = pickle.load(f)

        with open(f'./hyperparameter_tuning/output/lr_scaler_seed_{SEED}.pkl', 'rb') as f:
            scaler = pickle.load(f)
        X_test  = X_test.fillna(0)
        X_test  = scaler.transform(X_test)

    if 'lasso' in file:
        with open(f'./hyperparameter_tuning/output/lasso_model_seed_{SEED}.pkl',  'rb') as f:
            model = pickle.load(f)

        with open(f'./hyperparameter_tuning/output/lasso_scaler_seed_{SEED}.pkl', 'rb') as f:
            scaler = pickle.load(f)
        X_test  = X_test.fillna(0)
        X_test  = scaler.transform(X_test)
        
    if 'ridge' in file:
        with open(f'./hyperparameter_tuning/output/ridge_model_seed_{SEED}.pkl',  'rb') as f:
            model = pickle.load(f)

        with open(f'./hyperparameter_tuning/output/ridge_scaler_seed_{SEED}.pkl', 'rb') as f:
            scaler = pickle.load(f)
        X_test  = X_test.fillna(0)
        X_test  = scaler.transform(X_test)
    
    if 'svr' in file:
        with open(f'./hyperparameter_tuning/output/svr_model_seed_{SEED}.pkl',  'rb') as f:
            model = pickle.load(f)

        with open(f'./hyperparameter_tuning/output/svr_scaler_seed_{SEED}.pkl', 'rb') as f:
            scaler = pickle.load(f)
        X_test  = X_test.fillna(0)
        X_test  = scaler.transform(X_test)
        
    # Avoid platform-dependent worker-spawning failures when a persisted
    # scikit-learn estimator was originally fitted with n_jobs=-1.
    if hasattr(model, "n_jobs"):
        model.n_jobs = 1

    y_test_pred  = np.clip(model.predict(X_test),  0, 100)
    
    r2   = r2_score(np.array(y_test), y_test_pred)
    rmse = root_mean_squared_error(np.array(y_test), y_test_pred)

    pred_results.append({
        'model_name': name,
        'y_pred': y_test_pred,
        'r2': r2,
        'rmse': rmse
    })

    print(f"{name}: R2 = {r2:.4f}, RMSE = {rmse:.4f}")

#%% Supplementary figures - Fig. 2 (All parity plots (Figure))

MM_TO_INCH = 1 / 25.4

FIG_WIDTH = 180 * MM_TO_INCH
FIG_HEIGHT = 170 * MM_TO_INCH

FS_PANEL = 8.0
FS_TITLE = 7.0
FS_LABEL = 6.5
FS_TICK = 5.5
FS_TEXT = 5.5

mpl.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': 'Arial',

    'font.size': FS_TEXT,
    'axes.labelsize': FS_LABEL,
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

y_test_arr = np.asarray(y_test).ravel()
lims = np.array([-5, 105])
major_ticks = [0, 20, 40, 60, 80, 100]

#Figure
fig, axes = plt.subplots(
    nrows=3,
    ncols=3,
    figsize=(FIG_WIDTH, FIG_HEIGHT),
    dpi=300,

    sharex=False,
    sharey=False
)

axes = axes.ravel()

for idx, result in enumerate(pred_results):

    if idx >= len(axes):
        break

    ax = axes[idx]

    y_pred = np.asarray(result['y_pred']).ravel()
    model_name = result['model_name']
    r2 = result['r2']
    rmse = result['rmse']

    panel_label = string.ascii_lowercase[idx]

    #Parity plot
    ax.scatter(
        y_test_arr,
        y_pred,
        color='#56B4E9',
        edgecolors='white',
        linewidths=0.35,
        s=15,
        alpha=0.85,
        zorder=3
    )

    ax.plot(
        lims,
        lims,
        color='black',
        linewidth=0.9,
        linestyle='--',
        zorder=2
    )

    ax.fill_between(
        lims,
        lims - 10,
        lims + 10,
        color='gray',
        alpha=0.12,
        linewidth=0,
        zorder=1
    )

    metric_text = (f'$R^2$ = {r2:.4f}\n'
                   f'RMSE = {rmse:.2f}')
    ax.text(
        0.045,
        0.955,
        metric_text,
        transform=ax.transAxes,
        fontsize=FS_TEXT,
        ha='left',
        va='top',
        linespacing=1.15,
        bbox={
            'boxstyle': 'round,pad=0.28',
            'facecolor': 'white',
            'edgecolor': '#CCCCCC',
            'linewidth': 0.5,
            'alpha': 0.9
        },
        zorder=5
    )

    ax.set_title(
    '',
    loc='center',
    fontsize=FS_TITLE,
    fontweight='bold',
    pad=3)

    # Panel label: subplot 왼쪽 상단
    ax.text(
        -0.02,
        1.035,
        panel_label,
        transform=ax.transAxes,
        fontsize=FS_PANEL,
        fontweight='bold',
        ha='left',
        va='bottom',
        clip_on=False
    )

    ax.set_xlim(*lims)
    ax.set_ylim(*lims)
    
    ax.set_xlabel(
    'Actual THFA yield (%)',
    fontsize=FS_LABEL,
    labelpad=2
    )
    
    ax.set_ylabel(
        'Predicted THFA yield (%)',
        fontsize=FS_LABEL,
        labelpad=2
    )

    ax.set_xticks(major_ticks)
    ax.set_yticks(major_ticks)

    ax.tick_params(
        axis='x',
        which='major',
        labelbottom=True,
        labelsize=FS_TICK,
        direction='out',
        length=2.5,
        width=0.6,
        pad=2
    )

    ax.tick_params(
        axis='y',
        which='major',
        labelleft=True,
        labelsize=FS_TICK,
        direction='out',
        length=2.5,
        width=0.6,
        pad=2
    )

    ax.set_aspect(
        'equal',
        adjustable='box'
    )

    # 네 방향이 닫힌 상자형 축
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(0.6)

    ax.set_axisbelow(True)

for idx in range(len(pred_results), len(axes)):
    axes[idx].set_visible(False)

fig.subplots_adjust(
    left=0.095,
    right=0.985,
    bottom=0.075,
    top=0.965,
    wspace=0.22,
    hspace=0.30
)

# plt.savefig('./figure_supp_2.png', dpi=600, bbox_inches='tight')
# plt.savefig('./figure_supp_2.pdf', dpi=600, bbox_inches='tight')
save_figure(fig, "figure_s02")  # noqa: F405
