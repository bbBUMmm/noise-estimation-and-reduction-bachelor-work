import cv2
import numpy as np

# Nacitanie obrazku
image = cv2.imread('../Obrazky/ObrazkyJPEG/macka.jpg')

# Generacia Gaussovho sumu
mean = 0
stddev = 25  # Adjust this for more/less noise
gaussian_noise = np.random.normal(mean, stddev, image.shape).astype(np.float32)

# Pridanie sumu
noisy_image = image.astype(np.float32) + gaussian_noise

# Orezanie hodnot a prevod
noisy_image = np.clip(noisy_image, 0, 255).astype(np.uint8)

# Ukladanie vysledku
cv2.imwrite('ObrazokSGaussovymSumom.jpg', noisy_image)
