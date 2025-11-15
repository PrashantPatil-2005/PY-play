"""
plot_honeypot.py
Creates a clean IEEE-style Honeypot Defense Architecture diagram (Figure 5).

Usage:
    python plot_honeypot.py
"""

import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

def box(ax, xy, w, h, text, fontsize=9):
    rect = FancyBboxPatch(
        (xy[0], xy[1]), w, h,
        boxstyle="round,pad=0.3,rounding_size=4",
        ec="black", fc="white", linewidth=1.2
    )
    ax.add_patch(rect)
    ax.text(xy[0] + w/2, xy[1] + h/2, text,
            ha="center", va="center", fontsize=fontsize, wrap=True)

def arrow(ax, start, end):
    arr = FancyArrowPatch(
        start, end,
        arrowstyle='-|>', mutation_scale=12,
        lw=1.2, color="black",
        shrinkA=5, shrinkB=5
    )
    ax.add_patch(arr)

def main(out="honeypot_architecture.png"):
    fig, ax = plt.subplots(figsize=(8.5, 5))
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 80)
    ax.axis("off")

    w, h = 28, 12

    # Row 1: File Upload → Monitor Directory
    box(ax, (5, 55), w, h, "User Uploads\n(Images / Files)")
    box(ax, (38, 55), w, h, "Monitored Directory\n(Watchdog + SHA-256 Hashing)")
    arrow(ax, (5+w, 61), (38, 61))

    # Row 2: Change Detection → Classifier
    box(ax, (70, 55), w, h, "Change Detection\n(File Modified / Added?)")
    arrow(ax, (38+w, 61), (70, 61))

    box(ax, (20, 30), w, h, "Steganalysis Model\n(DeepStegDetect)")
    arrow(ax, (70+ w/2, 55), (20 + w/2, 42))

    # Row 3: Decision
    box(ax, (60, 30), w, h, "Classification Output\nCover / Stego")
    arrow(ax, (20+w, 36), (60, 36))

    # Row 4: Alerts / Dashboard
    box(ax, (38, 8), w, h, "Alert System\nEmail / Dashboard Notification")
    arrow(ax, (60 + w/2, 30), (38 + w/2, 20))

    # Caption
    ax.text(50, 3,
            "Figure 5: Honeypot-based Active Steganography Detection Architecture.",
            ha="center", fontsize=9)

    plt.tight_layout()
    plt.savefig(out, dpi=300, bbox_inches='tight')
    print("Saved:", out)


if __name__ == "__main__":
    main()
