"""
draw_pipeline_ieee_final.py
FLAWLESS IEEE-STYLE FLOWCHART (Straight lines only — NO diagonal arrows)
"""

import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

def box(ax, x, y, w, h, text):
    rect = FancyBboxPatch((x, y), w, h,
                          boxstyle="round,pad=0.35,rounding_size=4",
                          linewidth=1.2, edgecolor="black", facecolor="white")
    ax.add_patch(rect)
    ax.text(x + w/2, y + h/2, text,
            ha="center", va="center", fontsize=9, wrap=True)

def arrow(ax, start, end):
    ax.add_patch(FancyArrowPatch(start, end,
                                 arrowstyle="-|>", mutation_scale=12,
                                 linewidth=1.2, color="black"))

def main():
    fig, ax = plt.subplots(figsize=(7, 8))
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 150)
    ax.axis("off")

    W, H = 60, 15
    X = 20

    Y = [130, 110, 90, 70, 50]   # Perfect vertical spacing

    # --- Main Vertical Flow ---
    labels = [
        "BOSSBase + Multi-Algorithm Stego\n(S-UNIWARD • WOW • HILL • HUGO)",
        "Preprocessing\n160×160 • RGB • Normalize",
        "SRM High-Pass Filters\n(3 Filters → 9 Channels)",
        "ResNet50 Transfer Learning\nFreeze (Conv1–L3)",
        "Training & Evaluation\nBCE • Adam • Acc/F1"
    ]

    for i in range(len(Y)):
        box(ax, X, Y[i], W, H, labels[i])
        if i > 0:
            arrow(ax, (X + W/2, Y[i] + H), (X + W/2, Y[i-1]))

    # --- PERFECT SPLIT (NO CURVES, NO DIAGONALS!) ---
    split_y_top = 50
    split_y_bottom = 40
    center_x = X + W/2

    # Vertical stem
    arrow(ax, (center_x, split_y_top), (center_x, split_y_bottom))

    # Bottom row boxes (left & right)
    BW, BH = 36, 15
    left_x = 5
    right_x = 100 - BW - 5
    bottom_y = 15

    # Draw bottom boxes
    box(ax, left_x, bottom_y, BW, BH, "XAI (Grad-CAM)\nAttention Maps")
    box(ax, right_x, bottom_y, BW, BH, "Honeypot Defense\nMonitor • Alerts")

    # LEFT ARROW — **pure vertical + horizontal**
    arrow(ax,
          (center_x, split_y_bottom),
          (left_x + BW/2, bottom_y + BH))

    # RIGHT ARROW — **pure vertical + horizontal**
    arrow(ax,
          (center_x, split_y_bottom),
          (right_x + BW/2, bottom_y + BH))

    ax.text(50, 5,
            "Figure 1: Pipeline ",
            ha="center", fontsize=9)

    plt.savefig("pipeline_ieee_final.png", dpi=400, bbox_inches="tight")
    print("Saved pipeline_ieee_final.png")

if __name__ == "__main__":
    main()
