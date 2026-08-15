"""Generate manuscript Figure 6d."""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from figures.common import *  # noqa: F403

#%% Supplementary figures - Fig. 19 (MSP sensitivity analysis)

plt.rcParams['font.family']        = 'sans-serif'
plt.rcParams['font.sans-serif']    = ['Arial']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['axes.linewidth']     = 0.8
plt.rcParams['pdf.fonttype']       = 42
plt.rcParams['ps.fonttype']        = 42

FS_AXIS = 7
FS_TICK = 6
FS_LEG  = 6.5
FS_BASE = 6  

# =========================
# Catalyst lifetime sensitivity data
# =========================
lifetime = np.array([0.5, 1, 2, 3])
msp_nire = [3193.84, 3178.83, 3171.33, 3168.83]
msp_pd   = [4986.36, 4019.23, 3535.67, 3374.16]
# =========================
# Line plot — single column 89 mm ≈ 3.5 in
# =========================
fig, ax = plt.subplots(figsize=(3.35, 2.9), dpi=300)


ax.axvline(1.0, color="0.55", linewidth=0.8, linestyle="--", zorder=0)

ax.plot(lifetime, msp_pd, marker="o", markersize=4.5, linewidth=1.0,
        color="#EE854A", label="5Pd/Al$_2$O$_3$", zorder=3)
ax.plot(lifetime, msp_nire, marker="*", markersize=7, linewidth=1.0,
        color="#2166AC", label="1Re-4Ni/Al$_2$O$_3$", zorder=3)

ax.set_xlabel("Catalyst lifetime (years)", fontsize=FS_AXIS, labelpad=4)
ax.set_ylabel(r"MSP (\$/ton THFA)", fontsize=FS_AXIS, labelpad=4)

ax.set_xticks(lifetime)
ax.tick_params(direction="in", length=3, width=0.8, labelsize=FS_TICK)

ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)


ax.text(1.0, ax.get_ylim()[1], " Baseline", ha="left", va="top",
        fontsize=FS_BASE, color="0.35")
ax.legend(frameon=False, fontsize=FS_LEG, loc="upper right",
          handlelength=1.5, handletextpad=0.5)

fig.tight_layout()

# plt.savefig('./figure_supp_19.png', dpi=600, bbox_inches='tight')
# plt.savefig('./figure_supp_19.pdf', dpi=600, bbox_inches='tight')
save_figure(fig, "figure_06d")  # noqa: F405
