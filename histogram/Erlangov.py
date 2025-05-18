import cv2
import numpy as np
import matplotlib.pyplot as plt

# Načítanie obrázku
image = cv2.imread('LennaTestImage.png')
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
image = image / 255.0  # Normalizácia

# Generovanie Erlangovho (Gamma) šumu
def generate_erlang_noise(image_shape, shape_param=2.0, scale_param=0.02):
    noise = np.random.gamma(shape=shape_param, scale=scale_param, size=image_shape)
    return noise

# Pridanie Erlangovho šumu
erlang_noise = generate_erlang_noise(image.shape, shape_param=2.0, scale_param=0.02)
noisy_image = image + erlang_noise
noisy_image = np.clip(noisy_image, 0, 1)

# Zobrazenie pôvodného a zašumeného obrázku
fig, axs = plt.subplots(1, 2, figsize=(12, 6))  # Väčšia veľkosť

axs[0].imshow(image)
axs[0].set_title('Pôvodný obrázok', fontsize=14)
axs[0].axis('off')

axs[1].imshow(noisy_image)
axs[1].set_title('Po pridaní Erlangovho (Gamma) šumu', fontsize=14)
axs[1].axis('off')

# Nastavenie rozloženia, aby sa texty nezrezávali
plt.subplots_adjust(top=0.88, bottom=0.05, left=0.05, right=0.95, wspace=0.15)
plt.show()
