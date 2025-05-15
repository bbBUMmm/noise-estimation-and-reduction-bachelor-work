import cv2
import numpy as np
from matplotlib import pyplot as plt


# Rewrote function formula for the Gaussian random variable distribution
def gaussian_probability(x, mu, sigma):
    coefficient = 1 / (sigma * np.sqrt(2 * np.pi))
    exponent = np.exp(-((x - mu) ** 2) / (2 * sigma ** 2))
    return coefficient * exponent


# Create a value range to sample noise from
x_values = np.arange(-50, 51)

mu = 5
sigma = 7
probabilities = gaussian_probability(x_values, mu, sigma)
probabilities /= probabilities.sum()

# Sample noise values
image_shape = (256, 256)
noise = np.random.choice(x_values, size=image_shape, p=probabilities)

# Create base image
image = np.full((256, 256), 128, dtype=np.uint8)

# # Draw square
# square_image = image.copy()
# square_image = cv2.rectangle(square_image, (64, 64), (192, 192), color=150, thickness=-1)
#
# # Draw circle
# circle_image = image.copy()
# circle_image = cv2.circle(circle_image, (128, 128), 50, color=200, thickness=-1)
#
# # Draw triangle
# triangle_image = image.copy()
# triangle_points = np.array([[128, 100], [100, 150], [156, 150]], np.int32)
# triangle_image = cv2.fillPoly(triangle_image, [triangle_points], color=235)

# Combine shapes
# final_image_without_noise = np.maximum(np.maximum(square_image, circle_image), triangle_image)

# final_image_without_noise_int = final_image_without_noise.astype(np.int16)
final_image_without_noise_int = image.astype(np.int16)

# Add noise
noisy_image = final_image_without_noise_int + noise.astype(np.int16)
noisy_image = np.clip(noisy_image, 0, 255).astype(np.uint8)

# Save image
cv2.imwrite('noisy_imageLabor.png', noisy_image)

# Display original noisy image and histogram
plt.subplot(1, 2, 1)
plt.title("Obrázok s Gausovským šumom")
plt.imshow(noisy_image, cmap='gray')
plt.axis('off')

plt.subplot(1, 2, 2)
plt.title("Histogram")
plt.hist(noisy_image.ravel(), bins=256, range=(0, 256), color='gray')
plt.xlabel("Intenzita pixlov")
plt.ylabel("Počet pixlov")

plt.tight_layout()
plt.show()


# ========================
# ✅ Added: Load saved image and compare visually + pixel check
# ========================
loaded_image = cv2.imread('noisy_imageLabor.png', cv2.IMREAD_GRAYSCALE)

plt.figure(figsize=(8, 4))
plt.subplot(1, 2, 1)
plt.title("Original (before saving)")
plt.imshow(noisy_image, cmap='gray')
plt.axis('off')

plt.subplot(1, 2, 2)
plt.title("Loaded from PNG")
plt.imshow(loaded_image, cmap='gray')
plt.axis('off')

plt.tight_layout()
plt.show()

# Print if they are pixel-identical
print("Identical ObrazkyJPEG?", np.array_equal(loaded_image, noisy_image))
