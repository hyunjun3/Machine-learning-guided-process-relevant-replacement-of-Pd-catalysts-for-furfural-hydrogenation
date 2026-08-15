"""Generate Supplementary Figure 5."""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from figures.common import *  # noqa: F403

globals().update(load_ml_context())  # noqa: F405


#%% Supplementary figures - Fig. 5-6 (covariate / boostrap calculation (Calculation))
X_train_float = X_train.astype(float) 
noble_metal = ['Pd', 'Pt', 'Rh', 'Ru', 'Ir']

Pd_only_df = X_train_float[(X_train_float['Pd'] > 0) & ((X_train_float[active_metal] > 0).sum(axis=1) == 1)].copy()

condition_1 = X_train_float['Ni'] > 0
condition_2 = X_train_float[noble_metal].sum(axis=1) == 0

Ni_only_df = X_train_float[condition_1 & condition_2].copy()

X_train_float = X_train.astype(float)

noble_metal = ['Pd', 'Pt', 'Rh', 'Ru', 'Ir']
noble_data = X_train_float[X_train_float[noble_metal].sum(axis=1) > 0]
non_noble = X_train_float[(X_train_float[noble_metal].sum(axis=1) == 0)]

family_df = pd.concat([Pd_only_df.assign(_family=1), Ni_only_df.assign(_family=0)], axis=0,ignore_index=True)
model_columns = list(X_train_float.columns)

metal_precursor_columns = precursor
metal_derived_columns = ['Substrate to metal ratio (mmol/mmol)']

exclude_from_balance = sorted(set(active_metal) | set(metal_precursor_columns) | set(metal_derived_columns))

def build_propensity_model(random_state=23, C=1.0):
    return Pipeline(
        steps=[('imputer', SimpleImputer(strategy='median', add_indicator=True)),
               ('scaler', StandardScaler()),
               ('logistic', LogisticRegression(penalty='l2', C=C, solver='lbfgs', max_iter=5000, random_state=random_state))
               ])

def estimate_overlap_weights(
    data,
    balance_columns,
    family_column='_family',
    random_state=23,
    C=1.0,
    propensity_clip=1e-4):

    X_cov = (data.loc[:, balance_columns].apply(pd.to_numeric, errors='coerce').replace([np.inf, -np.inf], np.nan))
    y_family = (data[family_column].astype(int).to_numpy())

    all_missing_columns = [column for column in X_cov.columns if X_cov[column].notna().sum() == 0]

    if all_missing_columns:
        X_cov = X_cov.drop(columns=all_missing_columns)

    propensity_model = build_propensity_model(random_state=random_state, C=C)
    propensity_model.fit(X_cov, y_family)
    propensity_score = (propensity_model.predict_proba(X_cov)[:, 1])
    propensity_score = np.clip(propensity_score, propensity_clip, 1.0 - propensity_clip)
    overlap_weight = np.where(y_family == 1, 1.0 - propensity_score, propensity_score)

    return {
        'propensity_score': propensity_score,
        'overlap_weight': overlap_weight,
        'propensity_model': propensity_model,
        'balance_columns_used': list(X_cov.columns),
        'all_missing_columns': all_missing_columns
    }

def effective_sample_size(weights):
    weights = np.asarray(weights, dtype=float)
    denominator = np.sum(weights ** 2)
    if denominator <= 0:
        return np.nan

    return (np.sum(weights) ** 2 / denominator)

def weighted_mean_and_variance(values, weights):
    values = np.asarray(values, dtype=float)
    weights = np.asarray(weights, dtype=float)
    valid = (np.isfinite(values) & np.isfinite(weights) & (weights >= 0))

    values = values[valid]
    weights = weights[valid]
    weight_sum = np.sum(weights)

    if weight_sum <= 0:
        return np.nan, np.nan

    mean = np.sum(weights * values) / weight_sum
    variance = np.sum(weights * (values - mean) ** 2) / weight_sum

    return mean, variance


def standardized_mean_difference(values, family, weights=None):

    values = np.asarray(values, dtype=float)
    family = np.asarray(family, dtype=int)

    if weights is None:
        weights = np.ones_like(values, dtype=float)
    else:
        weights = np.asarray(weights, dtype=float)

    pd_mask = family == 1
    ni_mask = family == 0

    pd_mean, pd_var = weighted_mean_and_variance(values[pd_mask], weights[pd_mask])
    ni_mean, ni_var = weighted_mean_and_variance(values[ni_mask], weights[ni_mask])
    pooled_sd = np.sqrt(0.5 * (pd_var + ni_var))

    if not np.isfinite(pooled_sd):
        return np.nan

    if pooled_sd < 1e-12:
        if np.isclose(pd_mean, ni_mean):
            return 0.0

        return np.inf * np.sign(pd_mean - ni_mean)

    return (pd_mean - ni_mean) / pooled_sd

def calculate_balance_table(data, balance_columns, overlap_weights, family_column='_family'):
    family = (data[family_column].astype(int).to_numpy())
    rows = []

    for column in balance_columns:
        raw_values = (pd.to_numeric(data[column], errors='coerce').replace([np.inf, -np.inf], np.nan))

        if raw_values.notna().sum() == 0:
            continue

        missing_indicator = (
            raw_values.isna()
            .astype(float)
            .to_numpy()
        )

        median_value = raw_values.median()

        imputed_values = (
            raw_values
            .fillna(median_value)
            .to_numpy(dtype=float)
        )

        # Balance of median-imputed covariate
        smd_before = standardized_mean_difference(
            values=imputed_values,
            family=family,
            weights=None
        )

        smd_after = standardized_mean_difference(
            values=imputed_values,
            family=family,
            weights=overlap_weights
        )

        # Balance of missingness itself
        missing_smd_before = standardized_mean_difference(
            values=missing_indicator,
            family=family,
            weights=None
        )

        missing_smd_after = standardized_mean_difference(
            values=missing_indicator,
            family=family,
            weights=overlap_weights
        )

        rows.append({
            'Covariate': column,

            'SMD_before': smd_before,
            'SMD_after': smd_after,

            'Abs_SMD_before': abs(smd_before),
            'Abs_SMD_after': abs(smd_after),

            'Missing_SMD_before': missing_smd_before,
            'Missing_SMD_after': missing_smd_after,

            'Abs_Missing_SMD_before': abs(
                missing_smd_before
            ),
            'Abs_Missing_SMD_after': abs(
                missing_smd_after
            ),

            'Missing_fraction': (
                missing_indicator.mean()
            )
        })

    return (
        pd.DataFrame(rows)
        .sort_values(
            'Abs_SMD_after',
            ascending=False
        )
        .reset_index(drop=True)
    )

def make_common_grid(pd_data, ni_data, feature, n_grid=100, quantile_range=(0.0, 1.0), hard_limit=None):
    q_low, q_high = quantile_range

    pd_values = (
        pd_data[feature]
        .astype(float)
        .dropna()
        .to_numpy()
    )

    ni_values = (
        ni_data[feature]
        .astype(float)
        .dropna()
        .to_numpy()
    )

    lower = max(
        np.quantile(pd_values, q_low),
        np.quantile(ni_values, q_low)
    )

    upper = min(
        np.quantile(pd_values, q_high),
        np.quantile(ni_values, q_high)
    )

    if hard_limit is not None:
        lower = max(
            lower,
            hard_limit[0]
        )

        upper = min(
            upper,
            hard_limit[1]
        )

    return np.linspace(
        lower,
        upper,
        n_grid
    )


def build_prediction_matrix(estimator, X_model, feature, grid):

    prediction_matrix = np.empty((len(X_model), len(grid)), dtype=float)
    X_modified = X_model.copy()

    for grid_idx, feature_value in enumerate(grid):
        X_modified.loc[:, feature] = float(
            feature_value
        )

        prediction_matrix[:, grid_idx] = (
            np.asarray(
                estimator.predict(X_modified),
                dtype=float
            ).reshape(-1)
        )

    return prediction_matrix


def calculate_weighted_family_curve(prediction_matrix, family, weights, target_family):
    family = np.asarray(family, dtype=int)
    weights = np.asarray(weights, dtype=float)
    mask = family == target_family

    return np.average(
        prediction_matrix[mask, :],
        axis=0,
        weights=weights[mask]
    )

def run_overlap_weighted_family_pdp(
    estimator,
    family_data,
    model_columns,
    features,
    x_limits=None,
    exclude_balance_columns=None,
    n_grid=100,
    quantile_range=(0.0, 1.0),
    n_bootstrap=500,
    confidence_level=0.95,
    propensity_C=1.0,
    random_state=23
):

    if x_limits is None:
        x_limits = {}

    if exclude_balance_columns is None:
        exclude_balance_columns = []

    rng = np.random.default_rng(random_state)
    family_data = family_data.copy().reset_index(drop=True)
    family = family_data['_family'].astype(int).to_numpy()
    pd_indices = np.flatnonzero(family == 1)
    ni_indices = np.flatnonzero(family == 0)

    X_model = family_data[model_columns].astype(float)
    results = {}
    diagnostic_rows = []

    alpha = 1.0 - confidence_level
    lower_quantile = alpha / 2.0
    upper_quantile = 1.0 - alpha / 2.0

    for feature_idx, feature in enumerate(features):
        print(
            f'Processing {feature} '
            f'({feature_idx + 1}/{len(features)})'
        )

        excluded = (set(exclude_balance_columns) | {feature})
        balance_columns = [column for column in model_columns
                           if (column not in excluded and family_data[column].nunique(dropna=False) > 1)]

        grid = make_common_grid(
            pd_data=family_data.loc[family_data['_family'] == 1],
            ni_data=family_data.loc[family_data['_family'] == 0],
            feature=feature, n_grid=n_grid,
            quantile_range=quantile_range,
            hard_limit=x_limits.get(feature, None)
        )

        prediction_matrix = build_prediction_matrix(
            estimator=estimator, X_model=X_model, feature=feature, grid=grid)

        weight_result = estimate_overlap_weights(
            data=family_data,
            balance_columns=balance_columns,
            family_column='_family',
            random_state=random_state + feature_idx,
            C=propensity_C
        )
        
        balance_columns = weight_result[
            'balance_columns_used'
        ]
        overlap_weights = weight_result[
            'overlap_weight'
        ]

        propensity_scores = weight_result[
            'propensity_score'
        ]

        pd_curve = calculate_weighted_family_curve(
            prediction_matrix=prediction_matrix,
            family=family,
            weights=overlap_weights,
            target_family=1
        )

        ni_curve = calculate_weighted_family_curve(
            prediction_matrix=prediction_matrix,
            family=family,
            weights=overlap_weights,
            target_family=0
        )

        difference_curve = (
            ni_curve - pd_curve
        )
        
        #Bootstrap
        pd_bootstrap_curves = []
        ni_bootstrap_curves = []

        for bootstrap_idx in range(n_bootstrap):
            sampled_pd_indices = rng.choice(
                pd_indices,
                size=len(pd_indices),
                replace=True
            )

            sampled_ni_indices = rng.choice(
                ni_indices,
                size=len(ni_indices),
                replace=True
            )

            bootstrap_indices = np.concatenate(
                [
                    sampled_pd_indices,
                    sampled_ni_indices
                ]
            )

            bootstrap_data = (
                family_data.iloc[
                    bootstrap_indices
                ]
                .reset_index(drop=True)
            )

            bootstrap_family = bootstrap_data[
                '_family'
            ].astype(int).to_numpy()

            try:
                bootstrap_weight_result = estimate_overlap_weights(
                    data=bootstrap_data,
                    balance_columns=balance_columns,
                    family_column='_family',
                    random_state=(
                        random_state
                        + feature_idx * 100_000
                        + bootstrap_idx
                    ),
                    C=propensity_C
                )
                
                bootstrap_weights = bootstrap_weight_result[
                    'overlap_weight'
                ]

                bootstrap_prediction_matrix = (
                    prediction_matrix[
                        bootstrap_indices,
                        :
                    ]
                )

                pd_bootstrap_curve = (
                    calculate_weighted_family_curve(
                        prediction_matrix=bootstrap_prediction_matrix,
                        family=bootstrap_family,
                        weights=bootstrap_weights,
                        target_family=1
                    )
                )

                ni_bootstrap_curve = (
                    calculate_weighted_family_curve(
                        prediction_matrix=bootstrap_prediction_matrix,
                        family=bootstrap_family,
                        weights=bootstrap_weights,
                        target_family=0
                    )
                )

                pd_bootstrap_curves.append(
                    pd_bootstrap_curve
                )

                ni_bootstrap_curves.append(
                    ni_bootstrap_curve
                )

            except (
                ValueError,
                FloatingPointError
            ):
                continue

        pd_bootstrap_curves = np.asarray(
            pd_bootstrap_curves,
            dtype=float
        )

        ni_bootstrap_curves = np.asarray(
            ni_bootstrap_curves,
            dtype=float
        )

        difference_bootstrap_curves = (
            ni_bootstrap_curves
            - pd_bootstrap_curves
        )

        pd_lower = np.quantile(
            pd_bootstrap_curves,
            lower_quantile,
            axis=0
        )

        pd_upper = np.quantile(
            pd_bootstrap_curves,
            upper_quantile,
            axis=0
        )

        ni_lower = np.quantile(
            ni_bootstrap_curves,
            lower_quantile,
            axis=0
        )

        ni_upper = np.quantile(
            ni_bootstrap_curves,
            upper_quantile,
            axis=0
        )

        difference_lower = np.quantile(
            difference_bootstrap_curves,
            lower_quantile,
            axis=0
        )

        difference_upper = np.quantile(
            difference_bootstrap_curves,
            upper_quantile,
            axis=0
        )

        probability_ni_above_pd = np.mean(
            difference_bootstrap_curves > 0,
            axis=0
        )

        balance_table = calculate_balance_table(
            data=family_data,
            balance_columns=balance_columns,
            overlap_weights=overlap_weights,
            family_column='_family'
        )

        pd_weights = overlap_weights[
            family == 1
        ]

        ni_weights = overlap_weights[
            family == 0
        ]

        diagnostic_rows.append(
            {
                'Feature': feature,
                'Grid_min': grid.min(),
                'Grid_max': grid.max(),
                'Number_of_balance_covariates': len(
                    balance_columns
                ),
                'Mean_abs_SMD_before': (
                    balance_table[
                        'Abs_SMD_before'
                    ].replace(
                        [np.inf, -np.inf],
                        np.nan
                    ).mean()
                ),
                'Mean_abs_SMD_after': (
                    balance_table[
                        'Abs_SMD_after'
                    ].replace(
                        [np.inf, -np.inf],
                        np.nan
                    ).mean()
                ),
                'Max_abs_SMD_before': (
                    balance_table[
                        'Abs_SMD_before'
                    ].replace(
                        [np.inf, -np.inf],
                        np.nan
                    ).max()
                ),
                'Max_abs_SMD_after': (
                    balance_table[
                        'Abs_SMD_after'
                    ].replace(
                        [np.inf, -np.inf],
                        np.nan
                    ).max()
                ),
                'Pd_effective_sample_size': (
                    effective_sample_size(
                        pd_weights
                    )
                ),
                'Ni_effective_sample_size': (
                    effective_sample_size(
                        ni_weights
                    )
                ),
                'Successful_bootstraps': len(
                    pd_bootstrap_curves
                ),
                'Propensity_min': propensity_scores.min(),
                'Propensity_max': propensity_scores.max()
            }
        )

        results[feature] = {
            'grid': grid,

            'pd_curve': pd_curve,
            'pd_lower': pd_lower,
            'pd_upper': pd_upper,

            'ni_curve': ni_curve,
            'ni_lower': ni_lower,
            'ni_upper': ni_upper,

            'difference_curve': difference_curve,
            'difference_lower': difference_lower,
            'difference_upper': difference_upper,

            'probability_ni_above_pd': (
                probability_ni_above_pd
            ),

            'propensity_score': propensity_scores,
            'overlap_weight': overlap_weights,

            'balance_columns': balance_columns,
            'balance_table': balance_table,

            'pd_bootstrap_curves': pd_bootstrap_curves,
            'ni_bootstrap_curves': ni_bootstrap_curves
        }

    diagnostic_summary = pd.DataFrame(
        diagnostic_rows
    )

    return results, diagnostic_summary

#%% Supplementary figures - Fig. 5
features = [
    'Operating_temp',
    'Operating_pressure',
    'Operating_time'
]

x_labels = [
    'Operating temperature (°C)',
    'Operating pressure (bar)',
    'Operating time (h)'
]

x_lims = {
    'Operating_temp': (50, 200),
    'Operating_pressure': (0, 40),
    'Operating_time': (0, 10)
}

x_ticks = {
    'Operating_temp': [50, 100, 150, 200],
    'Operating_pressure': [0, 10, 20, 30, 40],
    'Operating_time': [0, 2, 4, 6, 8, 10]
}

balanced_pdp_results, balance_summary = (
    run_overlap_weighted_family_pdp(
        estimator=model,
        family_data=family_df,
        model_columns=model_columns,
        features=features,
        x_limits=x_lims,
        exclude_balance_columns=exclude_from_balance,
        n_grid=100,
        quantile_range=(0.0, 1.0),
        n_bootstrap=500,

        confidence_level=0.95,
        propensity_C=1.0,
        random_state=23
    )
)

pd.set_option('display.max_columns', None)
pd.set_option('display.width', 200)

n_pd = int((family_df['_family'] == 1).sum())
n_ni = int((family_df['_family'] == 0).sum())

balance_check = balance_summary.copy()

balance_check['Pd_original_n'] = n_pd
balance_check['Ni_original_n'] = n_ni

balance_check['Pd_ESS_ratio'] = (
    balance_check['Pd_effective_sample_size']
    / n_pd
)

balance_check['Ni_ESS_ratio'] = (
    balance_check['Ni_effective_sample_size']
    / n_ni
)

diagnostic_columns = [
    'Feature',
    'Grid_min',
    'Grid_max',
    'Number_of_balance_covariates',
    'Mean_abs_SMD_before',
    'Mean_abs_SMD_after',
    'Max_abs_SMD_before',
    'Max_abs_SMD_after',
    'Pd_effective_sample_size',
    'Pd_ESS_ratio',
    'Ni_effective_sample_size',
    'Ni_ESS_ratio',
    'Successful_bootstraps',
    'Propensity_min',
    'Propensity_max'
]

MM_TO_INCH = 1 / 25.4

FIG_WIDTH = 180 * MM_TO_INCH
FIG_HEIGHT = 102 * MM_TO_INCH

COLOR_PD = '#D95F5F'
COLOR_NI = '#4C78A8'

FS_AXIS = 7.5
FS_TICK = 7.0
FS_TITLE = 7.5
FS_LEGEND = 7.0
FS_PANEL = 8.5

LW_AXIS = 0.65
LW_LINE = 1.0

rcParams.update({
    'font.family': 'Arial',
    'font.size': FS_AXIS,

    'axes.linewidth': LW_AXIS,
    'axes.labelsize': FS_AXIS,
    'axes.titlesize': FS_TITLE,

    'xtick.labelsize': FS_TICK,
    'ytick.labelsize': FS_TICK,

    'legend.fontsize': FS_LEGEND,

    'pdf.fonttype': 42,
    'ps.fonttype': 42
})

pd_label = 'Pd-only catalysts'
ni_label = 'Ni-based non-noble metal catalysts'

bin_edges = np.linspace(0, 1, 31)

#Figure
fig, axes = plt.subplots(2, 3, figsize=(FIG_WIDTH, FIG_HEIGHT))

for idx, feature in enumerate(features):
    result = balanced_pdp_results[feature]
    propensity = np.asarray(result['propensity_score'], dtype=float)
    weights = np.asarray(result['overlap_weight'], dtype=float)
    family = (family_df['_family'].astype(int).to_numpy())

    valid = (np.isfinite(propensity) & np.isfinite(weights))
    propensity_valid = propensity[valid]
    weights_valid = weights[valid]
    family_valid = family[valid]
    pd_mask = family_valid == 1
    ni_mask = family_valid == 0

    ax = axes[0, idx]
    ax.hist(
        propensity_valid[pd_mask],
        bins=bin_edges,
        density=True,
        histtype='step',
        linewidth=LW_LINE,
        color=COLOR_PD,
        label=pd_label
    )
    ax.hist(
        propensity_valid[ni_mask],
        bins=bin_edges,
        density=True,
        histtype='step',
        linewidth=LW_LINE,
        color=COLOR_NI,
        label=ni_label
    )
    
    ax.set_xlim(0, 1)
    
    if idx == 0:
        ax.set_ylabel('Density', labelpad=3)
    else:
        ax.set_ylabel('')

    ax = axes[1, idx]
    ax.hist(
        propensity_valid[pd_mask],
        bins=bin_edges,
        weights=weights_valid[pd_mask],
        density=True,
        histtype='step',
        linewidth=LW_LINE,
        color=COLOR_PD,
        label=pd_label
    )
    ax.hist(
        propensity_valid[ni_mask],
        bins=bin_edges,
        weights=weights_valid[ni_mask],
        density=True,
        histtype='step',
        linewidth=LW_LINE,
        color=COLOR_NI,
        label=ni_label
    )

    ax.set_xlim(0, 1)

    ax.set_xlabel(
        'Propensity score for\nmonometallic Pd catalysts',
        labelpad=3
    )

    if idx == 0:
        ax.set_ylabel('Density', labelpad=3)
    else:
        ax.set_ylabel('')

panel_labels = ['a', 'b', 'c', 'd', 'e', 'f']

for panel_idx, ax in enumerate(axes.ravel()):

    ax.text(
        -0.16,
        1.06,
        panel_labels[panel_idx],
        transform=ax.transAxes,
        fontsize=FS_PANEL,
        fontweight='bold',
        va='top',
        ha='left'
    )

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    ax.spines['left'].set_linewidth(LW_AXIS)
    ax.spines['bottom'].set_linewidth(LW_AXIS)

    ax.tick_params(
        axis='both',
        which='major',
        direction='out',
        length=2.5,
        width=0.6,
        pad=2
    )

    ax.grid(False)

legend_handles = [
    Line2D(
        [0], [0],
        color=COLOR_PD,
        linewidth=2.0,
        label='Monometallic Pd catalysts'
    ),
    Line2D(
        [0], [0],
        color=COLOR_NI,
        linewidth=2.0,
        label='Ni-based non-noble catalysts'
    )
]

fig.legend(
    handles=legend_handles,
    loc='upper center',
    bbox_to_anchor=(0.53, 0.95),
    ncol=2,
    frameon=False,
    handlelength=2.5,
    columnspacing=1.8,
    handletextpad=0.6
)

fig.subplots_adjust(
    left=0.095,
    right=0.99,
    bottom=0.13,
    top=0.85,
    wspace=0.27,
    hspace=0.34
)

# plt.savefig('./figure_supp_5.png', dpi=600, bbox_inches='tight')
# plt.savefig('./figure_supp_5.pdf', dpi=600, bbox_inches='tight')
save_figure(fig, "figure_s05")  # noqa: F405
