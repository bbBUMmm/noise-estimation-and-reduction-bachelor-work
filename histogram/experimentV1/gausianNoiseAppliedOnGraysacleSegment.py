import csv
import cv2
import numpy as np
from matplotlib import pyplot as plt
import random
import os

# Rewrote function formula for the Gaussian random variable distribution
def gaussian_probability(x, mu, sigma):
    # 1. Coefficient part: 1 / (σ * sqrt(2π))
    coefficient = 1 / (sigma * np.sqrt(2 * np.pi))

    # 2. Exponent part: e^(-(x - μ)^2 / (2σ^2))
    exponent = np.exp(-((x - mu) ** 2) / (2 * sigma ** 2))

    # 3. Full formula
    return coefficient * exponent

# Load segment of the image
image = cv2.imread('../../ObrazkyPNG/VyrezyAleboSegmentyObrazkovPNG/greenSignPngCut.png', cv2.IMREAD_GRAYSCALE)

print(image)

# Getting metrix of image
height, width = image.shape

# Create a value range to sample noise from (centered around 0) with noise range -50 to 50
# This means that the maximum change the pixel can get is +-50 in intensity levev
x_values = np.arange(-50, 51)

# Get the probability for each x using formula and
# Randomly generate mu and sigma
mu = 0
sigma = random.uniform(5, 20)
probabilities = gaussian_probability(x_values, mu, sigma)

#
# Save mu and sigma to a CSV file so later we could check it
#
file_path = 'random_mu_sigma.csv'

# Check if file exists to write header only once
file_exists = os.path.isfile(file_path)

with open(file_path, mode='a', newline='') as file:
    writer = csv.writer(file)
    if not file_exists:
        writer.writerow(['mu', 'sigma'])
    writer.writerow([mu, sigma])

# Normalization to 1 because in valid probability distribution,
# the total probability over all possible outcomes must equal 1
probabilities /= probabilities.sum()

# Sample noise values based on the distribution
image_shape = (height, width)
noise = np.random.choice(x_values, size=image_shape, p=probabilities)

# Convert ObrazkyJPEG
final_image_without_noise_int = image.astype(np.int16)

noisy_image = final_image_without_noise_int + noise.astype(np.int16)

# Ensuring that all pixel values stay between 0 and 255, which is required for valid grayscale ObrazkyJPEG representation
noisy_image = np.clip(noisy_image, 0, 255).astype(np.uint8)

for index in range(0, 100):
    filename = f'noisy_image{index}.png'
    if not os.path.isfile(filename):
        cv2.imwrite(filename, noisy_image)
        break

# Result actions
plt.subplot(1, 2, 1)
plt.title("Obrázok s Gausovským šumom")
plt.imshow(noisy_image, cmap='gray')
plt.axis('off')

plt.subplot(1, 2, 2)
plt.title("Histogram")
plt.hist(noisy_image.ravel(), bins=256, range=(0, 256), color='gray')
plt.xlabel("Intenzita pixlov")
plt.ylabel("Poxet pixlov")

plt.tight_layout()
plt.show()

