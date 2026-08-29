"""Shared matplotlib style for all paper figures — one system, print-clean, colorblind-safe.
Import and call `apply()` at the top of each figure script. Outputs vector PDF.
"""
import matplotlib
matplotlib.use("pdf")
import matplotlib.pyplot as plt

# Colorblind-safe (Wong/Okabe-Ito) palette
BLUE   = "#0072B2"
ORANGE = "#E69F00"
GREEN  = "#009E73"
RED    = "#D55E00"
PURPLE = "#CC79A7"
GREY   = "#666666"
# diverging (heatmap): red (worse) -> white -> blue (better)
DIVERGING = "RdBu"

# single-column ACL width ~3.15in; double-column ~6.5in
COL_W = 3.15
DBL_W = 6.5


def apply():
    plt.rcParams.update({
        "figure.dpi": 150,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.02,
        "pdf.fonttype": 42,          # embed TrueType (editable, no Type-3 warnings)
        "ps.fonttype": 42,
        "font.family": "serif",       # match the paper body (Times-like)
        "font.size": 8,
        "axes.titlesize": 8,
        "axes.labelsize": 8,
        "xtick.labelsize": 7,
        "ytick.labelsize": 7,
        "legend.fontsize": 7,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.alpha": 0.25,
        "grid.linewidth": 0.5,
        "axes.axisbelow": True,
        "lines.linewidth": 1.6,
        "legend.frameon": False,
    })
