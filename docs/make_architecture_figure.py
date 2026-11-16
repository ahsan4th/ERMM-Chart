"""Fig. 13 of the article: architecture of the ERMM-Chart designer."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

BLUE, LBLUE, GREEN, LGREEN, ORANGE, LORANGE, GREY = "#1f4e79", "#dce8f5", "#1e6b3c", "#dff0e4", "#9c5a00", "#fbecd5", "#555555"
fig, ax = plt.subplots(figsize=(7.1, 4.3), dpi=300)
ax.set_xlim(0, 100); ax.set_ylim(0, 62); ax.axis("off")


def box(x, y, w, h, text, fc, ec, size=6.6, bold=False, ha="center"):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.4,rounding_size=1.2",
                                fc=fc, ec=ec, lw=0.9))
    ax.text(x + (w / 2 if ha == "center" else 1.2), y + h / 2, text, ha=ha, va="center",
            fontsize=size, fontweight="bold" if bold else "normal", color="black", linespacing=1.25)


def arrow(p, q, text="", color=GREY, rad=0.0, tx=None, ty=None, size=5.8):
    ax.add_patch(FancyArrowPatch(p, q, arrowstyle="-|>", mutation_scale=8, lw=0.9, color=color,
                                 connectionstyle=f"arc3,rad={rad}"))
    if text:
        ax.text(tx if tx is not None else (p[0] + q[0]) / 2, ty if ty is not None else (p[1] + q[1]) / 2,
                text, fontsize=size, color=color, ha="center", va="center", style="italic",
                bbox=dict(fc="white", ec="none", pad=0.6))


# column titles
for x, t, c in [(11, "Product design", GREEN), (50, "ERMM-Chart designer (web application)", BLUE),
                (89, "Finish-mill operation", ORANGE)]:
    ax.text(x, 60, t, ha="center", va="center", fontsize=7.6, fontweight="bold", color=c)

# product design
box(1, 41, 20, 14, "Product standard and\nplant targets\n(SNI 2049, strength\nclass, setting time)", LGREEN, GREEN)
box(1, 22, 20, 12, "Specification limits\nLSL, USL of each\ncharacteristic", LGREEN, GREEN)
box(1, 4, 20, 12, "Capability and\nproduct performance\n(feedback to design)", LGREEN, GREEN)
arrow((11, 41), (11, 34.6))

# tabs
tabs = ["\u2460 Product & process: data, specifications, action plan",
        "\u2461 Reference period: classical vs Fast-MCD estimates",
        "\u2462 Chart design: limit for N, p; critical shifts; choice of \u03bb",
        "\u2463 Monitoring: ERMM statistic, signals, production stops",
        "\u2464 Diagnosis & action: |U|, |V|, characteristics, units"]
y0 = 49.5
for i, t in enumerate(tabs):
    box(27.5, y0 - i * 7.6, 45, 5.6, t, LBLUE, BLUE, size=5.9, ha="left")
box(27.5, 3.2, 45, 9.5, "Computational core (Python, NumPy, SciPy)\nreweighted Fast-MCD (= robustbase) \u00b7 Max-M, EWMA, MEWMA\nMonte Carlo calibration (Alg. 1) \u00b7 ARL with CRN (Alg. 2)\nspecification-driven design (Eqs. 8, 9)",
    "#eef2f7", GREY, size=5.6)
arrow((50, 12.9), (50, 18.6), color=GREY)
ax.text(51.5, 15.6, "functions", fontsize=5.6, color=GREY, style="italic", ha="left", va="center")

# finish mill
box(80, 41, 19, 14, "Daily composite sample\nand laboratory: Blaine,\nresidue, free lime, LOI,\nSO\u2083; setting, strength", LORANGE, ORANGE)
box(80, 15, 19, 19, "Process units\nseparator, mill load,\ngrinding charge,\ngypsum feeder,\nkiln, clinker and\ncement storage", LORANGE, ORANGE)

# flows
arrow((21.4, 29.5), (26.3, 36.6), "critical\nshifts", color=GREEN, tx=23.4, ty=39.6)
arrow((21.4, 46), (26.3, 51.2), "limits", color=GREEN, tx=23.8, ty=53.4)
arrow((79.6, 50), (72.9, 52.2), "daily\nresults", color=ORANGE, tx=76.3, ty=54.6)
arrow((79.6, 45), (72.9, 29.0), color=ORANGE)
arrow((72.9, 21.6), (79.6, 24), "first\ncheck", color=BLUE, tx=76.3, ty=19.6)
arrow((26.6, 21.0), (21.6, 11.5), "properties,\ncapability", color=BLUE, tx=22.6, ty=18.6)
fig.savefig("fig13_architecture.png", bbox_inches="tight", pad_inches=0.03)
print("saved")
