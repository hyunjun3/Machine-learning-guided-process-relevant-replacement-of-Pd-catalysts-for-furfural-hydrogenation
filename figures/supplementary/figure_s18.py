"""Generate Supplementary Figure 18."""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from figures.common import *  # noqa: F403

#%% Supplementary figures - Fig. 18 (Tornado sensitivity analysis)
# =========================
# Style — Nature figure spec
# =========================
plt.rcParams['font.family']        = 'sans-serif'     
plt.rcParams['font.sans-serif']    = ['Arial']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['axes.linewidth']     = 0.8
plt.rcParams['pdf.fonttype']       = 42
plt.rcParams['ps.fonttype']        = 42

FS_LABEL = 8      
FS_AXIS  = 7      
FS_TICK  = 6     
FS_DATA  = 5.5    
FS_LEG   = 6.5   
FS_BASE  = 6.5   

# =========================
# Sensitivity data
# =========================
data = {
    "Pd": {
        "base": 4019.23,
        "FF price":       {"low": 3581.30, "high": 4457.26},
        "Catalyst price": {"low": 3825.80, "high": 4212.66},
        "H$_2$ price":    {"low": 3943.43, "high": 4095.03},
        "Utility cost":   {"low": 3962.90, "high": 4075.56},
        "IPA price":      {"low": 3983.62, "high": 4054.84},
    },
    "Ni-Re": {
        "base": 3178.83,
        "FF price":       {"low": 2727.22, "high": 3630.45},
        "Catalyst price": {"low": 3175.41, "high": 3181.41},
        "H$_2$ price":    {"low": 3100.41, "high": 3256.41},
        "Utility cost":   {"low": 3117.41, "high": 3239.91},
        "IPA price":      {"low": 3142.11, "high": 3215.41},
    }
}

PARAM_ORDER = [
    "FF price",
    "Catalyst price",
    "H$_2$ price",
    "Utility cost",
    "IPA price",
]
# =========================
# Tornado plot
# =========================
def plot_tornado(ax, case_name, case_data, letter):
    base = case_data["base"]

    rows = []
    for param, values in case_data.items():
        if param == "base":
            continue
        low_delta  = values["low"]  - base
        high_delta = values["high"] - base
        rows.append({
            "Parameter": param,
            "Low MSP":   values["low"],
            "High MSP":  values["high"],
            "Low delta":  low_delta,
            "High delta": high_delta,
            "Range": abs(high_delta - low_delta)
        })

    df = (pd.DataFrame(rows)
            .set_index("Parameter")
            .loc[PARAM_ORDER[::-1]]
            .reset_index())
    y = np.arange(len(df))

    ax.barh(y, df["Low delta"],  color="#4C78A8", alpha=0.90, height=0.62, label="-20%")
    ax.barh(y, df["High delta"], color="#F58518", alpha=0.90, height=0.62, label="+20%")
    ax.axvline(0, ymax=0.86, color="black", linewidth=0.9)

    ax.set_yticks(y)
    ax.set_yticklabels(df["Parameter"], fontsize=FS_TICK)
    ax.tick_params(axis="x", direction="in", length=3, width=0.8, labelsize=FS_TICK)
    ax.tick_params(axis="y", length=0)    

    ax.set_xlabel(r"MSP deviation from base case (\$/ton THFA)",
                  fontsize=FS_AXIS, labelpad=4)
    ax.set_title(letter, loc="left", fontsize=FS_LABEL, fontweight="bold", pad=6)

    ax.text(0, len(df) - 1 + 0.7,
            rf"{case_name}   Base MSP = {base:,.0f} \$/ton THFA",
            ha="center", va="bottom", fontsize=FS_BASE)
    x_span = max(abs(df["Low delta"]).max(), abs(df["High delta"]).max())
    offset = x_span * 0.04
    for i, row in df.iterrows():
        ax.text(row["Low delta"]  - offset, i, f"{row['Low MSP']:,.0f}",
                va="center", ha="right", fontsize=FS_DATA)
        ax.text(row["High delta"] + offset, i, f"{row['High MSP']:,.0f}",
                va="center", ha="left",  fontsize=FS_DATA)

    ax.set_xlim(-x_span * 1.34, x_span * 1.34)
    ax.set_ylim(-0.7, len(df) - 1 + 1.1)          
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    return df

fig, axes = plt.subplots(1, 2, figsize=(7.09, 3.3), dpi=300)

df_pd   = plot_tornado(axes[0], "5Pd",     data["Pd"],    "a")
df_nire = plot_tornado(axes[1], "1Re-4Ni", data["Ni-Re"], "b")

handles, labels = axes[0].get_legend_handles_labels()
fig.legend(handles, labels, loc="upper center", ncol=2, frameon=False,
           fontsize=FS_LEG, bbox_to_anchor=(0.5, 1.02),
           handlelength=1.2, handletextpad=0.5, columnspacing=1.5)

fig.tight_layout(rect=[0, 0, 1, 0.93], w_pad=1.5)

# plt.savefig('./figure_supp_18.png', dpi=600, bbox_inches='tight')
# plt.savefig('./figure_supp_18.pdf', dpi=600, bbox_inches='tight')
save_figure(fig, "figure_s18")  # noqa: F405
