"""
plot_loss.py
Generates IEEE-style Training Loss vs Epoch plot.
Outputs: loss_curve.png and loss_curve.pdf
"""

import matplotlib.pyplot as plt

# ---------------------------
# Training Loss Values
# ---------------------------
loss_values = [
    0.6942, 0.6933, 0.6932, 0.6925, 0.6927,
    0.6925, 0.6920, 0.6924, 0.6925, 0.6926,
    0.6924, 0.6918, 0.6919, 0.6917, 0.6916
]

epochs = list(range(1, len(loss_values) + 1))

# ---------------------------
# Plotting (IEEE Clean Style)
# ---------------------------
plt.figure(figsize=(6, 4))  # good IEEE size

plt.plot(
    epochs,
    loss_values,
    color='black',
    linewidth=1.8,
    marker='o',
    markersize=4
)

plt.xlabel("Epoch", fontsize=11)
plt.ylabel("Training Loss", fontsize=11)
plt.title("Training Loss vs Epoch", fontsize=12)

plt.grid(True, linestyle='--', linewidth=0.4)
plt.tight_layout()

# ---------------------------
# Save PNG + PDF
# ---------------------------
plt.savefig("loss_curve.png", dpi=400, bbox_inches='tight')
plt.savefig("loss_curve.pdf", dpi=400, bbox_inches='tight')

plt.close()
print("Saved: loss_curve.png and loss_curve.pdf")
