import cv2
import numpy as np
from matplotlib import pyplot as plt

# Parameters for Rayleigh distribution
sigma = 10
size = (256, 256)

# Generate Rayleigh-distributed noise
# Note: np.random.rayleigh gives values >= 0
noise = np.random.rayleigh(scale=sigma, size=size).astype(np.float32)

# Normalize the noise to fit better into pixel value range
# Shift it so the average noise isn't too high
noise = noise - noise.min()
noise = (noise / noise.max()) * 30  # scale max to ~20 intensity range
noise = noise.astype(np.int16)

# Base ObrazkyJPEG with geometric shapes
image = np.full(size, 128, dtype=np.uint8)

# Square
square_image = image.copy()
square_image = cv2.rectangle(square_image, (64, 64), (192, 192), color=150, thickness=-1)

# Circle
circle_image = image.copy()
circle_image = cv2.circle(circle_image, (128, 128), 50, color=200, thickness=-1)

# Triangle
triangle_image = image.copy()
triangle_points = np.array([[128, 100], [100, 150], [156, 150]], np.int32)
triangle_image = cv2.fillPoly(triangle_image, [triangle_points], color=235)

# Combine shapes
base_image = np.maximum(np.maximum(square_image, circle_image), triangle_image)
base_image = base_image.astype(np.int16)

# Add Rayleigh noise
noisy_image = base_image + noise
noisy_image = np.clip(noisy_image, 0, 255).astype(np.uint8)

# Plotting
plt.figure(figsize=(10, 4))

plt.subplot(1, 2, 1)
plt.title("Obrázok s Rayleighovým šumom")
plt.imshow(noisy_image, cmap='gray')
plt.axis('off')

plt.subplot(1, 2, 2)
plt.title("Histogram")
plt.hist(noisy_image.ravel(), bins=256, range=(0, 256), color='gray')
plt.xlabel("Intenzita pixlov")
plt.ylabel("Počet pixlov")

plt.tight_layout()
plt.show()
