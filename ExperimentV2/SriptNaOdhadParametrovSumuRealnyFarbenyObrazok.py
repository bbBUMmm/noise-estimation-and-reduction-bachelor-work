import cv2
import numpy as np
from matplotlib import pyplot as plt
import os

# Load the noisy color image
noisy_image = cv2.imread('./ZasumenyRealnyObrazokPozityVHistograme_0.png', cv2.IMREAD_COLOR)
if noisy_image is None:
    raise FileNotFoundError("Image not found!")

# Convert BGR to RGB for correct display in matplotlib
noisy_image_rgb = cv2.cvtColor(noisy_image, cv2.COLOR_BGR2RGB)

# Channel labels
channel_labels = ['R', 'G', 'B']
colors = ['r', 'g', 'b']

# Prepare output for mean and std per channel
for i, color in enumerate(channel_labels):
    channel = noisy_image_rgb[:, :, i]

    # Compute histogram
    hist = cv2.calcHist([channel], [0], None, [256], [0, 256])
    total_pixels = channel.size
    normalized_hist = hist / total_pixels

    # Estimate mean (μ)
    estimated_mu = np.sum([i * normalized_hist[i] for i in range(256)])

    # Estimate variance
    estimated_variance = np.sum([(i - estimated_mu) ** 2 * normalized_hist[i] for i in range(256)])

    # Estimate std deviation (σ)
    estimated_sigma = np.sqrt(estimated_variance)

    print(f"{color}-channel:")
    print(f"  Estimated mean (μ): {estimated_mu:.2f}")
    print(f"  Estimated variance (σ²): {estimated_variance:.2f}")
    print(f"  Estimated std deviation (σ): {estimated_sigma:.2f}")
    print()

# Plot noisy image and histograms
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.title("Zašumený obrázok (RGB)")
plt.imshow(noisy_image_rgb)
plt.axis('off')

plt.subplot(1, 2, 2)
plt.title("Odhadnutý histogram pre všetky kanály")
for i, color in enumerate(colors):
    plt.hist(noisy_image_rgb[:, :, i].ravel(), bins=256, range=(0, 256), color=color, alpha=0.5, label=color.upper())

plt.xlabel("Intenzita pixelov")
plt.ylabel("Počet pixelov")
plt.legend()

# Save the histogram
output_dir = "Histogramy"
os.makedirs(output_dir, exist_ok=True)
plt.savefig(os.path.join(output_dir, "OdhadnutyHistogramRealnehoObrazkuSSumomColor.png"))

plt.tight_layout()
plt.show()
