"""Fig. 1 — Saturation diagnosis: fast BLiMP (saturates ~70M) vs masked train loss (keeps falling).
Data from checkpoints/gptbert_small_v1 eval + train logs (see PROJECT_MEMORY / §4)."""
import os
from plotstyle import apply, BLUE, RED, GREY, COL_W
apply()
import matplotlib.pyplot as plt

# fast BLiMP by words-seen (M)  — from eval_chck_*M_fast.log
words   = [1,2,3,4,5,6,7,8,9,10,20,30,40,50,60,70,80,90,100]
blimp   = [54.21,53.31,52.69,54.22,55.84,57.25,58.19,58.35,59.58,59.61,
           64.39,64.67,66.26,68.74,68.93,70.13,69.72,70.05,70.01]
# masked training loss: mean over a +-0.5M-word window at each milestone (train.log)
loss_w  = [5,10,15,20,25,30,35,40,45,50,55,60,65,70,75,80,85,90,95,98.5]
loss    = [3.853,3.433,3.328,2.917,2.870,2.842,2.675,2.710,2.488,2.531,2.394,2.378,2.419,2.237,2.160,2.194,2.180,2.039,2.199,2.194]

fig, ax1 = plt.subplots(figsize=(COL_W, 2.25))
ax1.plot(words, blimp, color=BLUE, marker="o", ms=3, label="fast BLiMP")
ax1.set_xlabel("Words seen (M)")
ax1.set_ylabel("fast BLiMP acc.", color=BLUE)
ax1.tick_params(axis="y", labelcolor=BLUE)
ax1.set_xscale("log")
ax1.set_xticks([1,3,10,30,100]); ax1.set_xticklabels(["1","3","10","30","100"])
ax1.set_ylim(50, 72)

# shaded "wasted" region 70M->100M (draw before points so it sits behind)
ax1.axvspan(70, 100, color=GREY, alpha=0.08, zorder=0)

# mark the saturation point (label low-left, arrow up-right; clears the loss line)
peak_x, peak_y = 70, 70.13
ax1.axvline(peak_x, color=GREY, ls=":", lw=0.8, zorder=1)
ax1.annotate("70.13 at 70M", xy=(peak_x, peak_y), xytext=(34, 71.0),
             fontsize=6.5, color=GREY, ha="center",
             arrowprops=dict(arrowstyle="->", color=GREY, lw=0.6))

ax2 = ax1.twinx()
ax2.spines["top"].set_visible(False)
ax2.plot(loss_w, loss, color=RED, marker="s", ms=3, ls="--", label="masked train loss")
ax2.set_ylabel("masked train loss", color=RED)
ax2.tick_params(axis="y", labelcolor=RED)
ax2.grid(False)

# "wasted tail" caption in the free bottom-right, nudged left of the shaded band edge
ax1.text(27, 53.2, "epochs 8--10:", fontsize=6, color=GREY, ha="center", style="italic")
ax1.text(27, 52.1, "fit, not competence", fontsize=6, color=GREY, ha="center", style="italic")

# combined legend in the open upper-left area
l1,lab1 = ax1.get_legend_handles_labels()
l2,lab2 = ax2.get_legend_handles_labels()
ax1.legend(l1+l2, lab1+lab2, loc="upper left", fontsize=6.5,
           bbox_to_anchor=(0.0, 0.86), borderaxespad=0.2)

out = os.path.join(os.path.dirname(__file__), "fig_saturation.pdf")
plt.savefig(out)
print("wrote", out)
