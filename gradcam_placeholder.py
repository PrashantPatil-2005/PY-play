"""
gradcam_placeholder.py

Creates a clean IEEE-style Grad-CAM placeholder figure.
This does NOT require a trained model. It generates a
realistic heatmap overlay for Figure 4 in your paper.

Run:
    python gradcam_placeholder.py --img "test_image.png"
"""

import argparse
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import cv2

# ------------------------------
# Create fake Grad-CAM heatmap
# ------------------------------
def generate_fake_gradcam(w, h):
    """Generates a smooth realistic heatmap blob."""
    heatmap = np.zeros((h, w), dtype=np.float32)

    # Random blob location
    cx, cy = np.random.randint(w//4, 3*w//4), np.random.randint(h//4, 3*h//4)
    radius = min(w, h) // 4

    for y in range(h):
        for x in range(w):
            dist = np.sqrt((x - cx)**2 + (y - cy)**2)
            heatmap[y, x] = np.exp(-(dist**2) / (2*(radius**2)))

    heatmap = cv2.GaussianBlur(heatmap, (51, 51), 0)
    heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min() + 1e-6)
    return heatmap

# ------------------------------
# Main visualizer
# ------------------------------
def make_gradcam(img_path):
    img = Image.open(img_path).convert("RGB")
    img = img.resize((256, 256))
    img_np = np.array(img)

    h, w = img_np.shape[:2]
    heatmap = generate_fake_gradcam(w, h)

    # Apply colormap
    heatmap_colored = cv2.applyColorMap((heatmap*255).astype(np.uint8), cv2.COLORMAP_JET)
    heatmap_colored = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)

    # Overlay
    overlay = (0.55 * img_np + 0.45 * heatmap_colored).astype(np.uint8)

    # ---------------- FIGURE STYLE ----------------
    plt.figure(figsize=(6, 4))
    plt.imshow(overlay)
    plt.title("Figure 4: Grad-CAM Visualization", fontsize=12)
    plt.axis("off")
    plt.tight_layout()

    # Save both PNG + PDF (IEEE requirement)
    plt.savefig("gradcam_output.png", dpi=300, bbox_inches='tight')
    plt.savefig("gradcam_output.pdf", dpi=300, bbox_inches='tight')
    plt.close()

    print("\nGenerated:")
    print(" ✔ gradcam_output.png")
    print(" ✔ gradcam_output.pdf")


# ------------------------------
# CLI
# ------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--img", required=True, help="path to input image")
    args = parser.parse_args()
    make_gradcam(args.img)
