"""Fig. 4 — Fragile-cluster heatmap: every lever's Delta vs the Phase-1 baseline on the
long-distance-dependency cluster. Red = destroyed, blue = improved. MPO is the only
row that improves the key filler-gap columns. Data from checkpoints/*eval*.log (fast BLiMP)."""
import os
import numpy as np
from plotstyle import apply, COL_W
apply()
import matplotlib.pyplot as plt

# canonical Phase-1 baseline (fast BLiMP subtask acc)
base = {"wh": 71.5, "ld": 32.5, "da": 62.0, "el": 91.5}
# per-lever fragile-cluster values (fast BLiMP), from the eval logs
lev = [
    ("Contrastive", {"wh": 10.5, "ld": 3.0,  "da": 19.0, "el": 63.5}),
    ("Muon",        {"wh": 25.0, "ld": 5.5,  "da": 27.0, "el": 65.0}),
    ("Span mask",   {"wh": 55.0, "ld": 9.0,  "da": 39.0, "el": 94.5}),
    ("SLM",         {"wh": 56.0, "ld": 12.5, "da": 37.5, "el": 90.5}),
    ("Morph tok.",  {"wh": 67.5, "ld": 23.0, "da": 41.5, "el": 93.5}),
    ("Adaptive",    {"wh": 70.0, "ld": 20.5, "da": 58.5, "el": 94.0}),
    ("Soup",        {"wh": 70.0, "ld": 30.0, "da": 61.0, "el": 91.5}),
    ("MPO (ours)", {"wh": 75.5, "ld": 36.5, "da": 59.5, "el": 91.5}),  # verified: v1 chck_dpo_1250 canary 2026-07-11
]
cols = ["wh/that\n+gap", "wh/that gap\n(long-dist.)", "distractor\nagr.", "ellipsis\n$n$-bar"]
ckeys = ["wh", "ld", "da", "el"]

M = np.array([[lv[k] - base[k] for k in ckeys] for _, lv in lev])
rows = [name for name, _ in lev]

fig, ax = plt.subplots(figsize=(COL_W, 2.9))
vmax = np.abs(M).max()
im = ax.imshow(M, cmap="RdBu", vmin=-vmax, vmax=vmax, aspect="auto")

ax.set_xticks(range(len(cols))); ax.set_xticklabels(cols, fontsize=6)
ax.set_yticks(range(len(rows))); ax.set_yticklabels(rows, fontsize=7)
# bold the ours row label
ax.get_yticklabels()[-1].set_fontweight("bold")
ax.tick_params(length=0)
ax.grid(False)

# annotate cells with the signed delta
for i in range(M.shape[0]):
    for j in range(M.shape[1]):
        v = M[i, j]
        txt = f"{v:+.0f}" if abs(v) >= 0.5 else "0"
        ax.text(j, i, txt, ha="center", va="center", fontsize=6,
                color="white" if abs(v) > vmax * 0.55 else "black")

# separate the ours row with a line
ax.axhline(len(rows) - 1.5, color="black", lw=0.8)

cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
cb.set_label("$\\Delta$ acc.\\ vs baseline", fontsize=6.5)
cb.ax.tick_params(labelsize=6)
ax.set_title("Long-distance dependency cluster", fontsize=7.5, pad=4)

out = os.path.join(os.path.dirname(__file__), "fig_fragile.pdf")
plt.savefig(out)
print("wrote", out)
