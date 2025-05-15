import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

# Pomocná funkcia na výpočet štatistiky a histogramu jedného kanála
def vypocitaj_statistiku(kanal, nazov):
    hist = cv2.calcHist([kanal], [0], None, [256], [0, 256])
    normalized_hist = hist / kanal.size

    mu = np.sum([i * normalized_hist[i] for i in range(256)])
    variance = np.sum([(i - mu) ** 2 * normalized_hist[i] for i in range(256)])
    sigma = np.sqrt(variance)

    print(f"\n--- {nazov} ---")
    print(f"Odhadovaná stredná hodnota (μ): {mu:.2f}")
    print(f"Odhadovaný rozptyl (σ²): {variance:.2f}")
    print(f"Odhadovaná smerodajná odchýlka (σ): {sigma:.2f}")

    return mu, sigma, hist

# Vytvorenie výstupného adresára
output_dir = "RGB_Histogramy"
os.makedirs(output_dir, exist_ok=True)

# Načítanie farebného obrázka a konverzia do RGB formátu
image_bgr = cv2.imread('ZasumenyRealnyObrazokPozityVHistograme_0.png')
if image_bgr is None:
    raise FileNotFoundError("Obrázok sa nenašiel.")
image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

# Inicializácia pomocných zoznamov
kanaly = ['R', 'G', 'B']
farby = ['red', 'green', 'blue']
mu_pred, sigma_pred, mu_po, sigma_po = [], [], [], []
smoothed_rgb = np.zeros_like(image_rgb)

# Vytvorenie grafu
plt.figure(figsize=(15, 9))

# Spracovanie každého farebného kanála osobitne
for i in range(3):
    kanal = image_rgb[:, :, i]

    # Výpočet štatistiky pred vyhladzovaním
    mu1, sigma1, hist_pred = vypocitaj_statistiku(kanal, f"{kanaly[i]}-kanál pred vyhladzovaním")
    mu_pred.append(mu1)
    sigma_pred.append(sigma1)

    # Automatická voľba veľkosti jadra podľa sigma
    ksize = int(6 * sigma1 + 1)
    if ksize % 2 == 0:
        ksize += 1

    # Vyhladenie obrázka Gaussovským filtrom (len raz)
    kanal_smoothed = cv2.GaussianBlur(kanal, (ksize, ksize), sigma1)
    smoothed_rgb[:, :, i] = kanal_smoothed

    # Výpočet štatistiky po vyhladzovaní
    mu2, sigma2, hist_po = vypocitaj_statistiku(kanal_smoothed, f"{kanaly[i]}-kanál po vyhladzovaní")
    mu_po.append(mu2)
    sigma_po.append(sigma2)

    # Zobrazenie histogramu pred vyhladením
    plt.subplot(3, 2, i * 2 + 1)
    plt.bar(range(256), hist_pred.ravel(), color=farby[i], width=1.0)
    plt.title(f"{kanaly[i]}-kanál: Histogram pred vyhladzovaním")
    plt.xlabel('Jas pixelov')
    plt.ylabel('Počet pixelov')
    plt.xlim([0, 255])

    # Zobrazenie histogramu po vyhladení
    plt.subplot(3, 2, i * 2 + 2)
    plt.bar(range(256), hist_po.ravel(), color=farby[i], width=1.0)
    plt.title(f"{kanaly[i]}-kanál: Histogram po vyhladzovaní")
    plt.xlabel('Jas pixelov')
    plt.ylabel('Počet pixelov')
    plt.xlim([0, 255])

# Vytlačenie parametrov šumu pred a po vyhladzovaní pre všetky kanály
print("\n========== Zhrnutie parametrov šumu ==========")
for i in range(3):
    print(f"{kanaly[i]}-kanál:")
    print(f"  Pred vyhladzovaním: μ = {mu_pred[i]:.2f}, σ = {sigma_pred[i]:.2f}")
    print(f"  Po vyhladzovaní:    μ = {mu_po[i]:.2f}, σ = {sigma_po[i]:.2f}")

# Uloženie výsledkov
smoothed_bgr = cv2.cvtColor(smoothed_rgb, cv2.COLOR_RGB2BGR)
cv2.imwrite(os.path.join(output_dir, "Vyhladeny_RGB_Obrazok.png"), smoothed_bgr)
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "RGB_Gauss_Histogramy_SK.png"), dpi=300)
plt.show()
