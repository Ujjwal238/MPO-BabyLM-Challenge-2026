"""Fig. 5 — (a) DPO training dynamics: preference accuracy + reward margin climb (the objective
works). (b) Eval plateau across saved checkpoints: conservative run (v1) sits flat above the
baseline and above the over-optimized run (v2) -> not cherry-picked, and conservatism wins.
Data: checkpoints/gptbert_small_dpo_v1/train.log + dpo_canary sweeps (see PROJECT_MEMORY)."""

import os
import numpy as np
import matplotlib.pyplot as plt
from plotstyle import apply, BLUE, ORANGE, RED, GREY, COL_W

apply()

# ------------------------------------------------------------
# (a) Per-step dynamics (v1)
# ------------------------------------------------------------
step = [1,100,200,300,400,500,600,700,800,900,1000,1100,1200,1300,1400,1500]
prefacc = [.500,.613,.688,.694,.775,.787,.787,.794,.762,.806,.850,.825,.838,.819,.794,.769]
margin = [0.83,0.22,1.47,1.01,4.47,5.90,4.83,3.65,7.36,3.76,5.99,4.56,6.66,6.71,5.97,3.73]

# ------------------------------------------------------------
# (b) Evaluation plateau
# ------------------------------------------------------------
ck = [750,1000,1250,1500]
v1_blimp = [70.22,70.28,70.24,70.22]

v2_ck = [1000,1250,1500]
v2_blimp = [70.13,70.14,70.17]

base_blimp = 70.01

# ------------------------------------------------------------
# Figure
# ------------------------------------------------------------
fig, (axA, axB) = plt.subplots(
    1,
    2,
    figsize=(COL_W * 2.05, 2.70),
)

fig.subplots_adjust(
    wspace=0.45,
    bottom=0.41,     # room for centered captions
)

# ============================================================
# (a) DPO dynamics
# ============================================================

axA.plot(
    step,
    prefacc,
    color=BLUE,
    marker="o",
    ms=2.5,
    label="pref. accuracy",
)

axA.set_xlabel("DPO step")
axA.set_ylabel("preference accuracy", color=BLUE)
axA.tick_params(axis="y", labelcolor=BLUE)
axA.set_ylim(0.45, 0.92)

axA.axhline(
    0.5,
    color=GREY,
    ls=":",
    lw=0.8,
)

axA.text(
    400,
    0.515,
    "chance",
    fontsize=6,
    color=GREY,
)

axA2 = axA.twinx()
axA2.spines["top"].set_visible(False)

axA2.plot(
    step,
    margin,
    color=ORANGE,
    marker="s",
    ms=2.5,
    ls="--",
    label="reward margin",
)

axA2.set_ylabel(
    "reward margin",
    color=ORANGE,
)

axA2.tick_params(
    axis="y",
    labelcolor=ORANGE,
)

axA2.grid(False)

l1, la1 = axA.get_legend_handles_labels()
l2, la2 = axA2.get_legend_handles_labels()

axA.legend(
    l1 + l2,
    la1 + la2,
    loc="lower right",
    fontsize=6,
    bbox_to_anchor=(1.0, 0.08)
)

# ============================================================
# (b) Evaluation plateau
# ============================================================

axB.plot(
    ck,
    v1_blimp,
    color=BLUE,
    marker="o",
    ms=3.5,
    label="conservative (ours)",
)

axB.plot(
    v2_ck,
    v2_blimp,
    color=RED,
    marker="^",
    ms=3.5,
    ls="--",
    label="over-optimized",
)

axB.axhline(
    base_blimp,
    color=GREY,
    ls=":",
    lw=1.0,
)

axB.text(
    770,
    base_blimp + 0.01,
    "baseline 70.01",
    fontsize=6,
    color=GREY,
)

axB.set_xlabel("checkpoint (DPO step)")
axB.set_ylabel("fast BLiMP")

axB.set_ylim(69.9, 70.4)
axB.set_xticks(ck)

axB.legend(
    loc="lower left",
    fontsize=6,
    bbox_to_anchor=(0.02, -0.04)
)

# ============================================================
# Centered captions below panels
# ============================================================

axA.text(
    0.5,
    -0.30,
    "(a) preference-phase dynamics",
    transform=axA.transAxes,
    ha="center",
    va="top",
    fontsize=7.5,
)

axB.text(
    0.5,
    -0.30,
    "(b) Evaluation plateau",
    transform=axB.transAxes,
    ha="center",
    va="top",
    fontsize=7.5,
)

# ------------------------------------------------------------
# Save
# ------------------------------------------------------------

out = os.path.join(
    os.path.dirname(__file__),
    "fig_dynamics.pdf",
)

plt.savefig(
    out,
    bbox_inches="tight",
)

print("wrote", out)