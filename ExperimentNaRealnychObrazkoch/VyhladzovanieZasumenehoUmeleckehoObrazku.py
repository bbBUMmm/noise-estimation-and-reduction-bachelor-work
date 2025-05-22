import cv2
import numpy as np
from matplotlib import pyplot as plt
import os

# Pomocná funkcia na výpočet štatistiky a histogramu
def vypocitaj_statistiku(obrazok, popis):
    hist = cv2.calcHist([obrazok], [0], None, [256], [0, 256])
    normalized_hist = hist / obrazok.size

    estimated_mu = np.sum([i * normalized_hist[i] for i in range(256)])
    estimated_variance = np.sum([(i - estimated_mu) ** 2 * normalized_hist[i] for i in range(256)])
    estimated_sigma = np.sqrt(estimated_variance)

    print(f"\n--- {popis} ---")
    print(f"Odhadovaná stredná hodnota (mu): {estimated_mu:.2f}")
    print(f"Odhadovaný rozptyl (sigma^2): {estimated_variance:.2f}")
    print(f"Odhadovaná štandardná odchýlka (sigma): {estimated_sigma:.2f}")

    return estimated_mu, estimated_sigma, hist

# Načítanie obrázka
noisy_image = cv2.imread('RealneObrazky/ZasumenyRealnyObrazokPozityVHistograme.png', cv2.IMREAD_GRAYSCALE)

# Výpočet štatistiky pred vyhladzovaním
mu_pred, sigma_pred, hist_pred = vypocitaj_statistiku(noisy_image, "Pred vyhladzovaním")

# Automatická voľba veľkosti jadra na základe sigma
ksize = int(6 * sigma_pred + 1)
if ksize % 2 == 0:
    ksize += 1

# Vyhladenie obrázka pomocou Gaussovského filtra
smoothed_image = noisy_image.copy()
for i in range(3):  # Trojnásobné vyhladzovanie ako v 1D verzii
    smoothed_image = cv2.GaussianBlur(smoothed_image, (ksize, ksize), sigma_pred)

# Výpočet štatistiky po vyhladzovaní
mu_po, sigma_po, hist_po = vypocitaj_statistiku(smoothed_image, "Po vyhladzovaní")

# Zobrazenie obrázkov a histogramov
plt.figure(figsize=(12, 6))

# Pôvodný obrázok
plt.subplot(2, 2, 1)
plt.imshow(noisy_image, cmap='gray')
plt.title('Zašumený obrázok')
plt.axis('off')

# Histogram pred vyhladzovaním
plt.subplot(2, 2, 2)
plt.bar(range(256), hist_pred.ravel(), color='gray', width=1.0)
plt.title('Odhadnutý Histogram')
plt.xlabel('Intenzita pixlov')
plt.ylabel('Počet pixlov')
plt.xlim([0, 255])

# Vyhladený obrázok
plt.subplot(2, 2, 3)
plt.imshow(smoothed_image, cmap='gray', vmin=0, vmax=255)
plt.title('Vyhladený obrázok')
plt.axis('off')

# Histogram po vyhladzovaní
plt.subplot(2, 2, 4)
plt.bar(range(256), hist_po.ravel(), color='gray', width=1.0)
plt.title('Histogram po vyhladzovaní')
plt.xlabel('Intenzita pixlov')
plt.ylabel('Počet pixlov')
plt.xlim([0, 255])

plt.tight_layout()

# Uloženie výstupu
output_dir = "Histogramy"
os.makedirs(output_dir, exist_ok=True)
plt.savefig(os.path.join(output_dir, "VyhladenieAStatistikaUmeleckGrayscaleObrazok.png"))
cv2.imwrite(os.path.join(output_dir, "VyhladenyObrazok.png"), smoothed_image)

plt.show()
