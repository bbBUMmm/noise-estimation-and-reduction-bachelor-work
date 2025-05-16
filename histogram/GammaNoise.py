import cv2
import numpy as np
from matplotlib import pyplot as plt

# Parametre pre Gamma (Erlangovo) rozdelenie
k = 5        # tvar (shape)
theta = 2    # mierka (scale)
size = (256, 256)

# Generovanie Gamma šumu
gamma_noise = np.random.gamma(shape=k, scale=theta, size=size).astype(np.float32)

# Normalizácia na rozsah [0, 30]
gamma_noise = gamma_noise - gamma_noise.min()
gamma_noise = (gamma_noise / gamma_noise.max()) * 30
gamma_noise = gamma_noise.astype(np.int16)

# Základný obrázok s geometrickými tvarmi
image = np.full(size, 128, dtype=np.uint8)

# Štvorec
square_image = image.copy()
square_image = cv2.rectangle(square_image, (64, 64), (192, 192), color=150, thickness=-1)

# Kruh
circle_image = image.copy()
circle_image = cv2.circle(circle_image, (128, 128), 50, color=200, thickness=-1)

# Trojuholník
triangle_image = image.copy()
triangle_points = np.array([[128, 100], [100, 150], [156, 150]], np.int32)
triangle_image = cv2.fillPoly(triangle_image, [triangle_points], color=235)

# Kombinácia všetkých tvarov
base_image = np.maximum(np.maximum(square_image, circle_image), triangle_image)
base_image = base_image.astype(np.int16)

# Pridanie Gamma šumu
noisy_image = base_image + gamma_noise
noisy_image = np.clip(noisy_image, 0, 255).astype(np.uint8)

# Vykreslenie obrázku a histogramu
plt.figure(figsize=(10, 4))

plt.subplot(1, 2, 1)
plt.title("Obrázok s Erlangovým (Gamma) šumom")
plt.imshow(noisy_image, cmap='gray')
plt.axis('off')

plt.subplot(1, 2, 2)
plt.title("Histogram")
plt.hist(noisy_image.ravel(), bins=256, range=(0, 256), color='gray')
plt.xlabel("Intenzita pixlov")
plt.ylabel("Počet pixlov")

plt.tight_layout()
plt.show()
