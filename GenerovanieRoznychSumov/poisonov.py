import numpy as np
import matplotlib.pyplot as plt
from PIL import Image


def poisson_noise(image):
    """
    Pridá Poissonov šum do obrázka.

    image: NumPy pole obrázka (V, Š, C)

    Vráti nový obrázok so šumom.
    """
    # Pre správny Poissonov šum potrebujeme pixelové hodnoty v rozmedzí [0,1]
    image_normalized = image / 255.0

    # Generovanie Poissonovho šumu
    noisy = np.random.poisson(image_normalized * 255) / 255.0

    # Prevod späť na rozsah 0-255 a typ uint8
    noisy_img = np.clip(noisy * 255, 0, 255).astype(np.uint8)

    return noisy_img


# Načítanie obrázka
img = Image.open("Inputs/LennaTestImage.png").convert("RGB")
img_np = np.array(img)

# Pridanie Poissonovho šumu
noisy_img = poisson_noise(img_np)

# Zobrazenie pôvodného a zašumeného obrázka
plt.figure(figsize=(14, 7))

plt.subplot(1, 2, 1)
plt.imshow(img_np)
plt.axis('off')
plt.title('Pôvodný obrázok', fontsize=16, pad=20)

plt.subplot(1, 2, 2)
plt.imshow(noisy_img)
plt.axis('off')
plt.title('Obrázok s Poissonovým šumom', fontsize=16, pad=20)

plt.subplots_adjust(top=0.85, bottom=0.05, left=0.05, right=0.95, wspace=0.15)
plt.show()
