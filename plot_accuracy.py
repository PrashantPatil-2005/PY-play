"""
plot_accuracy.py
Generates IEEE-style Training Accuracy vs Epoch plot.
Outputs: acc_curve.png and acc_curve.pdf
"""

import matplotlib.pyplot as plt

# ---------------------------
# Training Accuracy Values
# ---------------------------
acc_values = [
    0.5077, 0.5133, 0.5191, 0.5152, 0.5185,
    0.5216, 0.5220, 0.5217, 0.5203, 0.5177,
    0.5192, 0.5227, 0.5218, 0.5228, 0.5214
]

epochs = list(range(1, len(acc_values) + 1))

# ---------------------------
# Plotting (IEEE Clean Style)
# ---------------------------
plt.figure(figsize=(6, 4))

plt.plot(
    epochs,
    acc_values,
    color='black',
    linewidth=1.8,
    marker='s',
    markersize=4
)

plt.xlabel("Epoch", fontsize=11)
plt.ylabel("Training Accuracy", fontsize=11)
plt.title("Training Accuracy vs Epoch", fontsize=12)

plt.grid(True, linestyle='--', linewidth=0.4)
plt.tight_layout()

# ---------------------------
# Save PNG + PDF
# ---------------------------
plt.savefig("acc_curve.png", dpi=400, bbox_inches='tight')
plt.savefig("acc_curve.pdf", dpi=400, bbox_inches='tight')

plt.close()
print("Saved: acc_curve.png and acc_curve.pdf")
