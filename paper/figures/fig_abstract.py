import os
import numpy as np
import matplotlib
matplotlib.use("pdf")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

CRAYON_BLUE = "#2b6cb0"
CRAYON_ORANGE = "#dd6b20"
CRAYON_GREEN = "#2f855a"
CRAYON_RED = "#c53030"
CRAYON_GREY = "#555555"
plt.rcParams.update({"pdf.fonttype": 42, "font.family": "sans-serif"})


def rbox(ax, xy, w, h, fc, ec, txt, fs=8):
    ax.add_patch(FancyBboxPatch(
        xy, w, h,
        boxstyle="round,pad=0.02,rounding_size=0.06",
        fc=fc, ec=ec, lw=1.6, mutation_aspect=1))
    ax.text(xy[0] + w / 2, xy[1] + h / 2,
            txt, ha="center", va="center",
            fontsize=fs, color=ec)


with plt.xkcd(scale=1.0, length=110, randomness=3):

    fig = plt.figure(figsize=(3.3, 3.9))

    axA = fig.add_axes([0.04, 0.60, 0.92, 0.36])
    axB = fig.add_axes([0.14, 0.09, 0.82, 0.42])

    # ==========================================================
    # (a) Hybrid backbone
    # ==========================================================
    axA.set_xlim(0, 20)
    axA.set_ylim(-0.7, 6)
    axA.axis("off")

    rbox(axA, (0.4, 2.1), 3.4, 1.7, "#f0f0f0", CRAYON_GREY, "embed.", 7)
    rbox(axA, (4.4, 1.9), 4.0, 2.1, "#dbeafe", CRAYON_BLUE, "shared\ntransf.", 7.5)
    rbox(axA, (9.0, 2.1), 3.2, 1.7, "#dcfce7", CRAYON_GREEN, "LM\nhead", 7.5)

    axA.annotate("", xy=(4.4, 2.95), xytext=(3.8, 2.95),
                 arrowprops=dict(arrowstyle="-|>", color="k", lw=1.3))
    axA.annotate("", xy=(9.0, 2.95), xytext=(8.4, 2.95),
                 arrowprops=dict(arrowstyle="-|>", color="k", lw=1.3))

    rbox(axA, (13.4, 3.5), 6.2, 1.5, "#eef2ff", CRAYON_BLUE, "masked  15/16", 7)
    rbox(axA, (13.4, 1.0), 6.2, 1.5, "#fff1f0", CRAYON_RED, "causal  1/16", 7)

    axA.annotate("", xy=(13.4, 4.1), xytext=(12.2, 3.2),
                 arrowprops=dict(arrowstyle="-|>", color=CRAYON_BLUE, lw=1.2))
    axA.annotate("", xy=(13.4, 1.7), xytext=(12.2, 2.7),
                 arrowprops=dict(arrowstyle="-|>", color=CRAYON_RED, lw=1.2))

    axA.text(16.5, 0.35, "used by MPO",
             ha="center", fontsize=6.5,
             color=CRAYON_RED, style="italic")

    axA.text(10.0, -0.35,
             "(a) Hybrid backbone",
             ha="center", va="top",
             fontsize=8, clip_on=False)

    # ==========================================================
    # (b) Story
    # ==========================================================
    x1 = np.linspace(0, 7, 60)
    y1 = 3.6 * (1 - np.exp(-x1 / 1.7))

    axB.plot(x1, y1, color=CRAYON_BLUE, lw=2.6, solid_capstyle="round")

    x2 = np.linspace(7, 10, 20)
    axB.plot(x2, np.full_like(x2, y1[-1]),
             color=CRAYON_BLUE, lw=2.6,
             ls=(0, (2, 2)))

    axB.text(1.4, 4.0, "pretraining",
             color=CRAYON_BLUE, fontsize=8.5)

    axB.text(8.7, y1[-1] - 0.7, "wasted\nepochs",
             color=CRAYON_GREY, fontsize=7,
             ha="center")

    axB.plot(7, y1[-1], marker="*", ms=14,
             color=CRAYON_ORANGE,
             mec="black", mew=0.7, zorder=5)

    axB.annotate("train to\nsaturation",
                 xy=(7, y1[-1]),
                 xytext=(3.4, 1.0),
                 fontsize=7,
                 color=CRAYON_ORANGE,
                 arrowprops=dict(arrowstyle="->",
                                 color=CRAYON_ORANGE,
                                 lw=1.3))

    xm = np.linspace(7, 9.6, 24)
    ym = y1[-1] + 0.9 * (1 - np.exp(-(xm - 7) / 0.8))

    axB.plot(xm, ym,
             color=CRAYON_GREEN,
             lw=2.6,
             solid_capstyle="round")

    axB.plot(9.6, ym[-1],
             marker="o", ms=8,
             color=CRAYON_GREEN,
             mec="black", mew=0.7,
             zorder=5)

    axB.annotate("MPO: prefer real\nover corrupted",
                 xy=(8.9, ym[9]),
                 xytext=(3.0, 5.0),
                 fontsize=7.5,
                 color=CRAYON_GREEN,
                 arrowprops=dict(arrowstyle="->",
                                 color=CRAYON_GREEN,
                                 lw=1.3))

    axB.text(9.9, ym[-1] + 0.1,
             "win",
             fontsize=7.5,
             color=CRAYON_GREEN,
             va="center")

    axB.set_xlim(-0.3, 12.0)
    axB.set_ylim(-0.8, 6.2)

    axB.set_xlabel("training budget", fontsize=8.5)
    axB.set_ylabel("competence", fontsize=8.5)

    axB.set_xticks([])
    axB.set_yticks([])

    axB.spines["top"].set_visible(False)
    axB.spines["right"].set_visible(False)

    axB.text(5.85, -1.85,
             "(b) Reallocating the wasted budget",
             ha="center", va="top",
             fontsize=8,
             clip_on=False)

out = os.path.join(os.path.dirname(__file__), "fig_abstract.pdf")
plt.savefig(out, bbox_inches="tight", pad_inches=0.05)
print("wrote", out)
