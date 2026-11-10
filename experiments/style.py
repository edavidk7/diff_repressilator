"""One figure style for every plot (after figures4papers: Arial, semantic palette, no top/right spines, frameless
legends, no grid, 300-dpi PDF/PNG). Importing this module applies it."""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

PALETTE = {"blue_main": "#0F4D92", "blue_secondary": "#3775BA", "green_1": "#DDF3DE", "green_3": "#8BCF8B",
           "red_1": "#F6CFCB", "red_strong": "#B64342", "neutral": "#CFCECE", "grey": "#767676", "ink": "#272727",
           "teal": "#42949E", "violet": "#9A4D8E", "gold": "#E0A100"}

# ours in the anchor blue, the competing methods in contrasting hues
METHOD_COLOUR = {"gradient": PALETTE["blue_main"], "laplace": PALETTE["blue_secondary"], "abc": PALETTE["red_strong"],
                 "magi": PALETTE["violet"], "pinn": PALETTE["teal"], "dga": PALETTE["grey"]}
# observed channels, close to their fluorescent protein
CHANNEL_COLOUR = {"lac": "#1B998B", "tet": "#4F9A3C", "cI": "#C2364A", "LacI-fusion": "#5C4DA8",
                  "TetR-fusion": PALETTE["gold"], "CI-fusion": "#D57EB1"}
RGB_REPORTERS = ["#C2364A", "#4F9A3C", "#3B6FC9"]  # mKate2, Venus, CFP of the improved repressilator

plt.rcParams.update({
    "font.family": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
    "font.size": 11,
    "axes.spines.right": False,
    "axes.spines.top": False,
    "axes.linewidth": 1.4,
    "axes.titleweight": "bold",
    "axes.titlelocation": "left",
    "legend.frameon": False,
    "savefig.dpi": 300,
    "savefig.facecolor": "white",
    "svg.fonttype": "none",
    "pdf.fonttype": 42,
})
