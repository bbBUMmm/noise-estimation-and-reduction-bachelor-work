import numpy as np
import matplotlib.pyplot as plt
import os

# Vytvoríme obraz
image = np.full((30, 30), 177, dtype=np.uint8)

# Vytvor priečinok, ak neexistuje
output_folder = "Histogramy"
os.makedirs(output_folder, exist_ok=True)

# Nastavenie cesty a názvu obrázku
output_path = os.path.join(output_folder, "HistogramNezasumenehoUmeleckehoObrazku.png")

# Vykreslenie a uloženie obrázka
plt.figure(figsize=(10, 4))

plt.subplot(1, 2, 1)
plt.title("Ideálny obrázok bez šumu")
plt.imshow(image, cmap='gray', vmin=0, vmax=255)
plt.axis('off')

plt.subplot(1, 2, 2)
plt.title("Histogram")
plt.hist(image.ravel(), bins=256, range=(0, 256), color='gray')
plt.xlabel("Hodnoty intenzity pixelov")
plt.ylabel("Počet pixlov tejto hodnoty")

plt.tight_layout()
plt.savefig(output_path, dpi=300)  # Uloženie do súboru
plt.show()
