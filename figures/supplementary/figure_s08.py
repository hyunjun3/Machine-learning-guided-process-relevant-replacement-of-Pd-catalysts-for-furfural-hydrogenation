"""Generate Supplementary Figure 8."""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from figures.common import *  # noqa: F403

#%% Supplementary figures - Fig. 8 (bootstrap stability (Calculation))

B = 200
BOOTSTRAP_SEED = 20260721
MODEL_SEED = 23

# True: calculate the bootstrap ensemble again
# False: load bootstrap_ensemble_results.npz when it already exists
RECALCULATE = False

DATA_DIR = "./dataset"
MODEL_DIR = "./hyperparameter_tuning/output"

X_TRAIN_PATH = f"{DATA_DIR}/ML_dataset_final_x_train.csv"
Y_TRAIN_PATH = f"{DATA_DIR}/ML_dataset_final_y_train.csv"
X_TEST_PATH = f"{DATA_DIR}/ML_dataset_final_x_test.csv"
Y_TEST_PATH = f"{DATA_DIR}/ML_dataset_final_y_test.csv"
MOLAR_MASS_PATH = f"{DATA_DIR}/molar_mass.pickle"

PARAM_PATH = f"{MODEL_DIR}/xgb_params_seed_{MODEL_SEED}.pkl"
MODEL_PATH = f"{MODEL_DIR}/xgb_model_seed_{MODEL_SEED}.json"

RESULT_PATH = "./bootstrap_output/bootstrap_ensemble_results.npz"
SUMMARY_PATH = "./bootstrap_output/positive_region_stability_summary.csv"

# =============================================================================
# Candidate and operating conditions
# =============================================================================
candidate_metals = ["Co", "Fe", "Cu", "Ca", "Zn", "Re"]
candidate_labels = [f"1{metal}-4Ni" for metal in candidate_metals]
thresholds = np.arange(-10.0, 11.0, 1.0)

temperature_range = np.arange(120.0, 201.0, 1.0)
pressure_range = np.arange(10.0, 40.5, 0.5)

precursor_map = {
    "Ca": "Ca(NO3)2",
    "Co": "Co(NO3)2",
    "Cu": "Cu(NO3)2",
    "Fe": "Fe(NO3)3",
    "Ni": "Ni(NO3)2",
    "Pd": "PdCl2",
    "Re": "NH4ReO4",
    "Zn": "Zn(NO3)2",
}

support = "Al2O3"
solvent = "2-propanol"
preparation = "wet impregnation"

solvent_amount = 40.0
calcination_temperature = 500.0
calcination_time = 6.0
reduction_temperature = 400.0
reduction_time = 4.0
operating_time = 6.0
stirring_rate = 700.0
catalyst_amount = 100.0
furfural_amount = 300.0


# =============================================================================
# Helper functions
# =============================================================================
def build_case(feature_names, molar_mass, active_metals):
    """Create one catalyst case over the temperature-pressure grid."""
    temperature_grid, pressure_grid = np.meshgrid(
        temperature_range,
        pressure_range,
        indexing="xy",
    )

    temperatures = temperature_grid.ravel()
    pressures = pressure_grid.ravel()

    case = pd.DataFrame(
        0.0,
        index=np.arange(len(temperatures)),
        columns=feature_names,
    )

    for metal, loading in active_metals.items():
        case[metal] = loading
        case[precursor_map[metal]] = 1.0

    case[support] = 100.0 - sum(active_metals.values())
    case[preparation] = 1.0
    case[solvent] = solvent_amount

    case["Stirring rate (rpm)"] = stirring_rate
    case["Catalyst amount (mg)"] = catalyst_amount
    case["Furfural (mg)"] = furfural_amount

    case["Operating_temp"] = temperatures
    case["Operating_pressure"] = pressures
    case["Operating_time"] = operating_time

    case["Calcination_temp"] = calcination_temperature
    case["Calcination_time"] = calcination_time
    case["Reduction_temp"] = reduction_temperature
    case["Reduction_time"] = reduction_time

    furfural_mmol = furfural_amount / molar_mass["Furfural"]

    metal_mmol = sum(
        catalyst_amount
        * (loading / 100.0)
        / molar_mass[metal]
        for metal, loading in active_metals.items()
    )

    case["Substrate to metal ratio (mmol/mmol)"] = (
        furfural_mmol / metal_mmol
    )
    case["Substrate concentration (mg/ml)"] = (
        furfural_amount / solvent_amount
    )

    return case, temperatures, pressures


def predict_delta(model, pd_case, candidate_cases):
    """Calculate candidate yield minus Pd yield over the operating grid."""
    pd_prediction = np.clip(
        model.predict(pd_case),
        0.0,
        100.0,
    )

    delta = []

    for label in candidate_labels:
        candidate_prediction = np.clip(
            model.predict(candidate_cases[label]),
            0.0,
            100.0,
        )
        delta.append(candidate_prediction - pd_prediction)

    return np.asarray(delta, dtype=np.float32)


# =============================================================================
# Load data and model information
# =============================================================================
x_train = pd.read_csv(X_TRAIN_PATH)
y_train = pd.read_csv(Y_TRAIN_PATH).iloc[:, 0]

x_test = pd.read_csv(X_TEST_PATH)
y_test = pd.read_csv(Y_TEST_PATH).iloc[:, 0]

with open(MOLAR_MASS_PATH, "rb") as file:
    molar_mass = pickle.load(file)

with open(PARAM_PATH, "rb") as file:
    xgb_params = pickle.load(file)


# =============================================================================
# Build Pd and candidate operating grids
# =============================================================================
pd_case, temperatures, pressures = build_case(
    feature_names=x_train.columns,
    molar_mass=molar_mass,
    active_metals={"Pd": 5.0},
)

candidate_cases = {}

for metal, label in zip(candidate_metals, candidate_labels):
    candidate_cases[label], _, _ = build_case(
        feature_names=x_train.columns,
        molar_mass=molar_mass,
        active_metals={
            "Ni": 4.0,
            metal: 1.0,
        },
    )

print(f"Training data: {x_train.shape}")
print(f"Test data: {x_test.shape}")
print(f"Operating grid: {len(pd_case):,} points")


# =============================================================================
# Load existing bootstrap result or calculate again
# =============================================================================
if os.path.exists(RESULT_PATH) and not RECALCULATE:
    result = np.load(RESULT_PATH)

    bootstrap_delta = result["delta_y"]
    test_r2 = result["test_r2"]
    test_rmse = result["test_rmse"]
    mean_gap = result["mean_gap"]
    fraction_by_threshold = result["fraction_by_threshold"]
    ranks_by_threshold = result["ranks_by_threshold"]
    nominal_delta = result["nominal_delta_y"]

    print(f"Loaded: {RESULT_PATH}")

else:
    # -------------------------------------------------------------------------
    # Nominal model
    # -------------------------------------------------------------------------
    nominal_model = XGBRegressor(
        random_state=MODEL_SEED,
        n_jobs=-1,
        **xgb_params,
    )
    nominal_model.load_model(MODEL_PATH)

    nominal_delta = predict_delta(
        model=nominal_model,
        pd_case=pd_case,
        candidate_cases=candidate_cases,
    )

    re_index = candidate_labels.index("1Re-4Ni")

    nominal_re_fraction = (
        np.mean(nominal_delta[re_index] >= 0.0)
        * 100.0
    )
    nominal_re_mean_gap = np.mean(
        nominal_delta[re_index]
    )

    print(
        f"Nominal Re-Ni fraction: {nominal_re_fraction:.3f}%"
    )
    print(
        f"Nominal Re-Ni mean ΔY: {nominal_re_mean_gap:.3f} pp"
    )

    # -------------------------------------------------------------------------
    # Bootstrap refits
    # -------------------------------------------------------------------------
    n_train = len(x_train)
    n_candidates = len(candidate_labels)
    n_grid = len(pd_case)

    bootstrap_delta = np.zeros(
        (B, n_candidates, n_grid),
        dtype=np.float32,
    )
    test_r2 = np.zeros(B)
    test_rmse = np.zeros(B)

    start_time = time.time()

    for bootstrap_index in range(B):
        rng = np.random.default_rng(
            BOOTSTRAP_SEED + bootstrap_index
        )

        sampled_rows = rng.integers(
            0,
            n_train,
            size=n_train,
        )

        model = XGBRegressor(
            random_state=BOOTSTRAP_SEED + bootstrap_index,
            n_jobs=-1,
            **xgb_params,
        )

        model.fit(
            x_train.iloc[sampled_rows],
            y_train.iloc[sampled_rows],
            verbose=False,
        )

        bootstrap_delta[bootstrap_index] = predict_delta(
            model=model,
            pd_case=pd_case,
            candidate_cases=candidate_cases,
        )

        test_prediction = np.clip(
            model.predict(x_test),
            0.0,
            100.0,
        )

        test_r2[bootstrap_index] = r2_score(
            y_test,
            test_prediction,
        )
        test_rmse[bootstrap_index] = np.sqrt(
            mean_squared_error(
                y_test,
                test_prediction,
            )
        )

        if (
            (bootstrap_index + 1) % 10 == 0
            or bootstrap_index + 1 == B
        ):
            elapsed_min = (time.time() - start_time) / 60.0

            print(
                f"Bootstrap {bootstrap_index + 1}/{B} | "
                f"RMSE={test_rmse[bootstrap_index]:.3f} | "
                f"elapsed={elapsed_min:.1f} min"
            )

    # -------------------------------------------------------------------------
    # Positive-region fractions and candidate rankings
    # -------------------------------------------------------------------------
    mean_gap = np.mean(
        bootstrap_delta,
        axis=2,
    )

    fraction_by_threshold = np.mean(
        bootstrap_delta[:, :, :, None]
        >= thresholds[None, None, None, :],
        axis=2,
    ) * 100.0

    ranks_by_threshold = np.zeros(
        fraction_by_threshold.shape,
        dtype=np.int16,
    )

    for bootstrap_index in range(B):
        for threshold_index in range(len(thresholds)):
            # Larger region fraction is better.
            # Mean ΔY is used only to resolve ties.
            order = np.lexsort(
                (
                    -mean_gap[bootstrap_index],
                    -fraction_by_threshold[
                        bootstrap_index,
                        :,
                        threshold_index,
                    ],
                )
            )

            ranks_by_threshold[
                bootstrap_index,
                order,
                threshold_index,
            ] = np.arange(
                1,
                len(candidate_labels) + 1,
            )

    np.savez_compressed(
        RESULT_PATH,
        delta_y=bootstrap_delta,
        test_r2=test_r2,
        test_rmse=test_rmse,
        mean_gap=mean_gap,
        fraction_by_threshold=fraction_by_threshold,
        ranks_by_threshold=ranks_by_threshold,
        thresholds=thresholds,
        candidate_labels=np.asarray(candidate_labels),
        temperatures=temperatures,
        pressures=pressures,
        nominal_delta_y=nominal_delta,
    )

    print(f"Saved: {RESULT_PATH}")


# =============================================================================
# Save a compact summary table
# =============================================================================
zero_index = np.flatnonzero(
    np.isclose(thresholds, 0.0)
)[0]

fractions_zero = fraction_by_threshold[
    :,
    :,
    zero_index,
]

ranks_zero = ranks_by_threshold[
    :,
    :,
    zero_index,
]

summary = []

for candidate_index, label in enumerate(candidate_labels):
    values = fractions_zero[:, candidate_index]

    summary.append(
        {
            "Catalyst": label,
            "Nominal fraction (%)": (
                np.mean(
                    nominal_delta[candidate_index] >= 0.0
                )
                * 100.0
            ),
            "Bootstrap median fraction (%)": np.median(values),
            "Bootstrap 2.5th fraction (%)": np.percentile(
                values,
                2.5,
            ),
            "Bootstrap 97.5th fraction (%)": np.percentile(
                values,
                97.5,
            ),
            "Non-zero region frequency (%)": np.mean(
                values > 0.0
            )
            * 100.0,
            "Rank-1 frequency (%)": np.mean(
                ranks_zero[:, candidate_index] == 1
            )
            * 100.0,
        }
    )

summary_df = pd.DataFrame(summary)
summary_df.to_csv(SUMMARY_PATH, index=False)
 
#%% Supplementary figures - Fig. 8 (bootstrap stability (Figure))

RESULT_PATH = "./bootstrap_output/bootstrap_ensemble_results.npz"

result = np.load(RESULT_PATH)

labels = result["candidate_labels"].astype(str).tolist()
thresholds = result["thresholds"].astype(float)
nominal_delta = result["nominal_delta_y"].astype(float)

zero_index = np.flatnonzero(
    np.isclose(thresholds, 0.0)
)[0]

fractions = result[
    "fraction_by_threshold"
][:, :, zero_index]

ranks = result[
    "ranks_by_threshold"
][:, :, zero_index]

bootstrap_count = fractions.shape[0]
re_index = labels.index("1Re-4Ni")

re_fraction = fractions[:, re_index]

nominal_fraction = (
    np.mean(nominal_delta[re_index] >= 0.0)
    * 100.0
)

median_fraction = np.median(re_fraction)

low_fraction, high_fraction = np.percentile(
    re_fraction,
    [2.5, 97.5],
)

zero_count = np.sum(
    re_fraction == 0.0
)

positive_counts = np.sum(
    fractions > 0.0,
    axis=0,
)

positive_frequency = (
    positive_counts
    / bootstrap_count
    * 100.0
)

rank1_counts = np.sum(
    ranks == 1,
    axis=0,
)

rank1_frequency = (
    rank1_counts
    / bootstrap_count
    * 100.0
)


# =============================================================================
# Figure settings
# =============================================================================
mpl.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": [
            "Arial",
            "Helvetica",
            "Liberation Sans",
            "DejaVu Sans",
        ],
        "font.size": 6.3,
        "axes.labelsize": 6.7,
        "xtick.labelsize": 5.5,
        "ytick.labelsize": 5.7,
        "legend.fontsize": 5.3,
        "axes.linewidth": 0.65,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    }
)


def style_axis(axis):
    axis.grid(False)

    axis.tick_params(
        direction="out",
        width=0.6,
        length=2.5,
        pad=2,
    )

    for spine in axis.spines.values():
        spine.set_linewidth(0.65)


def panel_label(axis, label):
    axis.text(
        0.0,
        1.04,
        label,
        transform=axis.transAxes,
        ha="left",
        va="bottom",
        fontsize=8.2,
        fontweight="bold",
    )


mm_to_inch = 1.0 / 25.4

fig = plt.figure(
    figsize=(
        180 * mm_to_inch,
        64 * mm_to_inch,
    ),
    dpi=300,
)

grid = fig.add_gridspec(
    1,
    3,
    width_ratios=[1.28, 1.0, 1.0],
    left=0.065,
    right=0.985,
    bottom=0.185,
    top=0.90,
    wspace=0.38,
)

ax_a = fig.add_subplot(grid[0, 0])
ax_b = fig.add_subplot(grid[0, 1])
ax_c = fig.add_subplot(grid[0, 2])

color_re = "#E69F00"
color_re_dark = "#9E5A00"
color_other = "#8EB3CC"
edge_color = "#3F3F3F"


# =============================================================================
# Panel a
# =============================================================================
bins = np.arange(-2.5, 102.6, 5.0)

counts, _, _ = ax_a.hist(
    re_fraction,
    bins=bins,
    color=color_re,
    edgecolor=edge_color,
    linewidth=0.4,
)

ymax = max(
    float(np.max(counts)),
    1.0,
)

ax_a.vlines(
    nominal_fraction,
    0,
    ymax * 1.06,
    color="#222222",
    linestyle="--",
    linewidth=1.0,
)

ax_a.vlines(
    median_fraction,
    0,
    ymax * 1.06,
    color=color_re_dark,
    linewidth=1.1,
)

bracket_y = ymax * 0.70

ax_a.hlines(
    bracket_y,
    low_fraction,
    high_fraction,
    color=edge_color,
    linewidth=0.8,
)

ax_a.vlines(
    [low_fraction, high_fraction],
    bracket_y - ymax * 0.035,
    bracket_y + ymax * 0.035,
    color=edge_color,
    linewidth=0.8,
)

ax_a.text(
    (low_fraction + high_fraction) / 2.0,
    bracket_y + ymax * 0.06,
    f"95% interval: {low_fraction:.0f}-{high_fraction:.0f}%",
    ha="center",
    va="bottom",
    fontsize=5.5,
)

ax_a.text(
    4.0,
    ymax * 0.50,
    f"Zero region in {zero_count}/{bootstrap_count} refits",
    ha="left",
    va="center",
    fontsize=5.5,
)

ax_a.set_xlim(-2.5, 102.5)
ax_a.set_ylim(0, ymax * 1.34)

ax_a.set_xlabel(
    "Re-Ni positive-region fraction (%)"
)
ax_a.set_ylabel(
    "Number of bootstrap refits"
)

ax_a.legend(
    handles=[
        Line2D(
            [0],
            [0],
            color="#222222",
            linestyle="--",
            linewidth=1.0,
            label=f"Nominal = {nominal_fraction:.1f}%",
        ),
        Line2D(
            [0],
            [0],
            color=color_re_dark,
            linewidth=1.1,
            label=f"Bootstrap median = {median_fraction:.1f}%",
        ),
    ],
    loc="upper right",
    frameon=False,
    handlelength=2.0,
)

panel_label(ax_a, "a")
style_axis(ax_a)


# =============================================================================
# Panel b
# =============================================================================
x = np.arange(len(labels))

colors = [
    color_re if label == "1Re-4Ni" else color_other
    for label in labels
]

ax_b.bar(
    x,
    positive_frequency,
    width=0.68,
    color=colors,
    edgecolor=edge_color,
    linewidth=0.4,
)

for index, (count, value) in enumerate(
    zip(
        positive_counts,
        positive_frequency,
    )
):
    ax_b.text(
        index,
        value + 1.4,
        f"{value:.1f}%\n({count}/{bootstrap_count})",
        ha="center",
        va="bottom",
        fontsize=5.2,
        fontweight=(
            "bold"
            if labels[index] == "1Re-4Ni"
            else "normal"
        ),
    )

ax_b.set_ylim(0, 61)
ax_b.set_yticks(
    [0, 10, 20, 30, 40, 50, 60]
)

ax_b.set_ylabel(
    "Refits with a non-zero\npositive region (%)"
)

ax_b.set_xticks(x)
ax_b.set_xticklabels(
    labels,
    rotation=38,
    ha="right",
    rotation_mode="anchor",
)

panel_label(ax_b, "b")
style_axis(ax_b)


# =============================================================================
# Panel c
# =============================================================================
ax_c.bar(
    x,
    rank1_frequency,
    width=0.68,
    color=colors,
    edgecolor=edge_color,
    linewidth=0.4,
)

for index, value in enumerate(rank1_frequency):
    ax_c.text(
        index,
        value + 2.0,
        f"{value:.1f}%",
        ha="center",
        va="bottom",
        fontsize=5.3,
        fontweight=(
            "bold"
            if labels[index] == "1Re-4Ni"
            else "normal"
        ),
    )

ax_c.text(
    0.03,
    0.95,
    "Ranked by region fraction;\nmean ΔY resolves ties",
    transform=ax_c.transAxes,
    ha="left",
    va="top",
    fontsize=5.2,
    color="#444444",
)

ax_c.set_ylim(0, 108)
ax_c.set_yticks(
    [0, 20, 40, 60, 80, 100]
)

ax_c.set_ylabel(
    "Rank-1 frequency (%)"
)

ax_c.set_xticks(x)
ax_c.set_xticklabels(
    labels,
    rotation=38,
    ha="right",
    rotation_mode="anchor",
)

panel_label(ax_c, "c")
style_axis(ax_c)

# plt.savefig('./figure_supp_8.png', dpi=600, bbox_inches='tight')
# plt.savefig('./figure_supp_8.pdf', dpi=600, bbox_inches='tight')
save_figure(fig, "figure_s08")  # noqa: F405
