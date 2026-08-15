"""Generate Supplementary Figure 17."""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from figures.common import *  # noqa: F403

#%% Supplementary figures - Fig. 17 (Process-level cost breakdown)
# ── Nature figure settings ────────────────────────────────
plt.rcParams["font.family"]        = "sans-serif"
plt.rcParams["font.sans-serif"]    = ["Arial"]
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["axes.linewidth"]     = 0.8
plt.rcParams["pdf.fonttype"]       = 42   
plt.rcParams["ps.fonttype"]        = 42

# Font sizes (pt) 
FS_LABEL = 8     
FS_AXIS  = 7      
FS_TICK  = 6     
FS_DATA  = 6     
FS_LEG   = 6.5   

cases = ["Ni-Re", "Pd"]

OPERATING_HOURS = 8000

raw_cost = pd.DataFrame({
    "Ni-Re": {
        "Furfural":  76_868_480,
        "H2":        13_304_808,
        "IPA":        6_249_975.68,
        "Catalyst":     510_720,
    },
    "Pd": {
        "Furfural":  76_868_480,
        "H2":        13_304_808,
        "IPA":        6_249_975.68,
        "Catalyst":  33_951_168,
    }
})

summary = pd.DataFrame({
    "THFA production (10$^3$ ton/yr)": {
        "Ni-Re": 68_875 / 1e3,
        "Pd":    71_027 / 1e3,
    },
    "Utility cost (M$/yr)": {
        "Ni-Re": 10_396_138 / 1e6,
        "Pd":     9_887_154 / 1e6,
    },
    "CAPEX (M$)": {
        "Ni-Re": 9_475_630 / 1e6,
        "Pd":     8_985_560 / 1e6,
    }
})

colors_raw = {
    "Furfural":  "#4C78A8",
    "H2":        "#F58518",
    "IPA":       "#B279A2",
    "Catalyst":  "#54A24B",
}
legend_labels = {
    "Furfural": "Furfural",
    "H2":       "H$_2$",
    "IPA":      "IPA",
    "Catalyst": "Catalyst",
}
colors_case = {
    "Ni-Re": "#0F7C80",
    "Pd":    "#B8BEC8",
}

fig = plt.figure(figsize=(7.09, 3.9), dpi=300)
gs = GridSpec(
    nrows=3, ncols=2, figure=fig,
    width_ratios=[1.3, 1.0],
    height_ratios=[1, 1, 1],
    wspace=0.30, hspace=0.85
)

ax_raw       = fig.add_subplot(gs[:, 0])
axes_summary = [fig.add_subplot(gs[i, 1]) for i in range(3)]

# ── Panel a ────────────────────────────────────────────────
x         = np.arange(len(cases))
bar_width = 0.55
bottom    = np.zeros(len(cases))
totals    = raw_cost[cases].sum(axis=0).values

for item in raw_cost.index:
    values = raw_cost.loc[item, cases].values.astype(float)
    ax_raw.bar(
        x, values / 1e6,
        bottom=bottom / 1e6,
        width=bar_width,
        color=colors_raw[item],
        edgecolor="white",
        linewidth=0.6,
        label=legend_labels[item]
    )
    percents = values / totals * 100
    for i, (v, p) in enumerate(zip(values, percents)):
        if p >= 3:
            ax_raw.text(
                x[i], (bottom[i] + v / 2) / 1e6,
                f"{p:.1f}%",
                ha="center", va="center",
                fontsize=FS_DATA, color="black"
            )
    bottom += values

for i, total in enumerate(totals / 1e6):
    ax_raw.text(
        x[i], total + 3.0,
        f"{total:.2f} M$/yr",
        ha="center", va="bottom",
        fontsize=FS_DATA
    )

ax_raw.set_ylabel("Raw material cost (M$/yr)", fontsize=FS_AXIS)
ax_raw.set_xticks(x)
ax_raw.set_xticklabels(["1Re-4Ni", "5Pd"])
ax_raw.set_ylim(0, max(totals / 1e6) * 1.18)
ax_raw.tick_params(direction="in", length=3, width=0.8, labelsize=FS_TICK)
ax_raw.spines["top"].set_visible(False)
ax_raw.spines["right"].set_visible(False)
ax_raw.legend(
    frameon=False, fontsize=FS_LEG, loc="upper left",
    bbox_to_anchor=(0.02, 0.98), handlelength=1.2, handletextpad=0.5
)
ax_raw.set_title("a", loc="left", fontsize=FS_LABEL, fontweight="bold", pad=4)

# ── Panels b-d ─────────────────────────────────────────────
panel_letters = ["b", "c", "d"]

for ax, metric, letter in zip(axes_summary, summary.columns, panel_letters):
    vals  = summary[metric].loc[cases].values
    y_pos = np.arange(len(cases))

    bars = ax.barh(
        y_pos, vals, height=0.5,
        color=[colors_case[c] for c in cases],
        edgecolor="black", linewidth=0.5
    )

    ax.set_yticks(y_pos)
    ax.set_yticklabels(["1Re-4Ni", "5Pd"])
    ax.invert_yaxis()

    for bar, v in zip(bars, vals):
        fmt = f"{v:.3f}" if metric == "CAPEX (M$)" else f"{v:.2f}"
        ax.text(
            bar.get_width() + max(vals) * 0.03,
            bar.get_y() + bar.get_height() / 2,
            fmt, ha="left", va="center", fontsize=FS_DATA
        )

    ax.set_xlabel(metric, fontsize=FS_AXIS)          # 지표명 → x축 라벨 (단위 포함)
    ax.set_xlim(0, max(vals) * 1.28)
    ax.tick_params(direction="in", length=3, width=0.8, labelsize=FS_TICK)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.set_title(letter, loc="left", fontsize=FS_LABEL, fontweight="bold", pad=4)

fig.tight_layout(pad=0.6, h_pad=0.8, w_pad=1.2)

# plt.savefig('./figure_supp_17.png', dpi=600, bbox_inches='tight')
# plt.savefig('./figure_supp_17.pdf', dpi=600, bbox_inches='tight')
save_figure(fig, "figure_s17")  # noqa: F405
