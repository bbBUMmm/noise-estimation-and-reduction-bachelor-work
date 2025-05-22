import cv2
import numpy as np
from matplotlib import pyplot as plt


# Načítajte zašumený obrázok (nahraďte cestu k vášmu zašumenému obrázku)
# noisy_image = cv2.imread('noisy_image0.png', cv2.IMREAD_GRAYSCALE)

# Skuska nad umelecky vytvorenym obrazkom
noisy_image = cv2.imread('noisy_image5.png', cv2.IMREAD_GRAYSCALE)

# Získajte Histogram zašumeného obrázka
hist = cv2.calcHist([noisy_image], [0], None, [256], [0, 256])

# Normalizujte Histogram, aby sme získali odhad pravdepodobnostnej distribúcie
total_pixels = noisy_image.size
normalized_hist = hist / total_pixels

# Inicializujte premenné pre výpočet strednej hodnoty a rozptylu
estimated_mu = 0
estimated_variance = 0

# Vypočítajte odhad strednej hodnoty
for i in range(256):
    estimated_mu += i * normalized_hist[i]

# Vypočítajte odhad rozptylu
for i in range(256):
    estimated_variance += (i - estimated_mu) ** 2 * normalized_hist[i]

# Vypočítajte odhad štandardnej odchýlky
estimated_sigma = np.sqrt(estimated_variance)

# Zobrazte odhadnuté parametre
print(f"Odhadovaná stredná hodnota (mu): {estimated_mu[0]:.2f}")
print(f"Odhadovaný rozptyl (sigma^2): {estimated_variance[0]:.2f}")
print(f"Odhadovaná štandardná odchýlka (sigma): {estimated_sigma[0]:.2f}")

# Zobrazte zašumený obrázok a jeho Histogram
plt.subplot(1, 2, 1)
plt.title("Zašumený obrázok")
plt.imshow(noisy_image, cmap='gray')
plt.axis('off')

plt.subplot(1, 2, 2)
plt.title("Odhadnutý Histogram")
plt.hist(noisy_image.ravel(), bins=256, range=(0, 256), color='gray')
plt.xlabel("Intenzita pixlov")
plt.ylabel("Počet pixlov")

plt.tight_layout()
plt.show()