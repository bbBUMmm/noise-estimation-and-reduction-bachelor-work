import cv2
import matplotlib.pyplot as plt
import numpy as np

# Load original and smoothed images
img_before = cv2.imread('./RealneObrazky/PolaroidPNGVyrez.png')
img_after = cv2.imread('./RGB_Histogramy/Vyhladeny_RGB_Obrazok.png')

# Convert BGR to RGB for matplotlib
img_before_rgb = cv2.cvtColor(img_before, cv2.COLOR_BGR2RGB)
img_after_rgb = cv2.cvtColor(img_after, cv2.COLOR_BGR2RGB)

# Function to calculate statistics per channel
def compute_stats(img, label):
    means = np.mean(img, axis=(0, 1))
    stds = np.std(img, axis=(0, 1))
    print(f"\nŠtatistiky pre {label}:")
    for i, color in enumerate(['R', 'G', 'B']):
        print(f"  {color}-kanál: μ = {means[i]:.2f}, σ = {stds[i]:.2f}")
    return means, stds

# Compute and print stats
compute_stats(img_before_rgb, "Pôvodný obrázok")
compute_stats(img_after_rgb, "Vyhladený obrázok")

# Plot only the two images side by side
fig, axs = plt.subplots(1, 2, figsize=(12, 6))
# fig.suptitle("Porovnanie pôvodného a vyhladeného obrázka", fontsize=16)

axs[0].imshow(img_before_rgb)
axs[0].set_title("Obrázok pred experimentom")
axs[0].axis('off')

axs[1].imshow(img_after_rgb)
axs[1].set_title("Obrázok po experimente a vyhladzovaní")
axs[1].axis('off')

plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.show()
