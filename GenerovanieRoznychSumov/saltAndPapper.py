import cv2
import numpy as np
from matplotlib import pyplot as plt

# Parametre pre impulzný šum („soľ a korenie“)
prob_salt = 0.05  # pravdepodobnosť soli (biela)
prob_pepper = 0.05  # pravdepodobnosť korenia (čierna)
size = (256, 256)

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

# Generovanie impulzného šumu
noisy_image = base_image.copy()
total_pixels = size[0] * size[1]

# Náhodné pozície pre soľ
num_salt = int(prob_salt * total_pixels)
coords_salt = (np.random.randint(0, size[0], num_salt), np.random.randint(0, size[1], num_salt))
noisy_image[coords_salt] = 255

# Náhodné pozície pre korenie
num_pepper = int(prob_pepper * total_pixels)
coords_pepper = (np.random.randint(0, size[0], num_pepper), np.random.randint(0, size[1], num_pepper))
noisy_image[coords_pepper] = 0

# Zobrazenie obrázka a histogramu
plt.figure(figsize=(10, 4))

plt.subplot(1, 2, 1)
plt.title("Obrázok s impulzným šumom (soľ a korenie)")
plt.imshow(noisy_image, cmap='gray')
plt.axis('off')

plt.subplot(1, 2, 2)
plt.title("Histogram")
plt.hist(noisy_image.ravel(), bins=256, range=(0, 256), color='gray')
plt.xlabel("Intenzita pixlov")
plt.ylabel("Počet pixlov")

plt.tight_layout()
plt.show()
