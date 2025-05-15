import cv2
import numpy as np
from matplotlib import pyplot as plt

# Load original and noisy ObrazkyJPEG
original_image = cv2.imread('../../ObrazkyPNG/VyrezyAleboSegmentyObrazkovPNG/greenSignPngCut.png', cv2.IMREAD_GRAYSCALE)
noisy_image = cv2.imread('noisy_image0.png', cv2.IMREAD_GRAYSCALE)

# Calculate the noise matrix
noise = noisy_image.astype(np.int16) - original_image.astype(np.int16)

# Filter noise to only values within [-50, 50]
filtered_noise = noise[(noise >= -50) & (noise <= 50)]

# Calculate histogram of noise values from -50 to 50
bins = np.arange(-50, 51)  # 101 bins for range [-50, 50]
hist, bin_edges = np.histogram(filtered_noise, bins=bins)

# Normalize histogram to get probability estimate
total_filtered = filtered_noise.size
normalized_hist = hist / total_filtered
x_values = bin_edges[:-1]  # x axis: -50 to 50

# Estimate mu and sigma
estimated_mu = np.sum(x_values * normalized_hist)
estimated_variance = np.sum((x_values - estimated_mu) ** 2 * normalized_hist)
estimated_sigma = np.sqrt(estimated_variance)

# Show estimated parameters
print(f"Odhadovaná stredná hodnota (mu): {estimated_mu:.2f}")
print(f"Odhadovaný rozptyl (sigma^2): {estimated_variance:.2f}")
print(f"Odhadovaná štandardná odchýlka (sigma): {estimated_sigma:.2f}")

# Plot histogram of noise
plt.subplot(1, 2, 1)
plt.title("Zašumený obrázok")
plt.imshow(noisy_image, cmap='gray')
plt.axis('off')

plt.subplot(1, 2, 2)
plt.title("Histogram šumu v rozsahu [-50, 50]")
plt.bar(x_values, hist, width=1, color='gray')
plt.xlabel("Hodnoty šumu")
plt.ylabel("Počet výskytov")

plt.tight_layout()
plt.show()
