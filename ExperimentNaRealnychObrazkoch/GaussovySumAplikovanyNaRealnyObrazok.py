import csv
import cv2
import numpy as np
from matplotlib import pyplot as plt
import random
import os

# Load the real color image (BGR format in OpenCV)
image = cv2.imread('./RealneObrazky/PolaroidPNGVyrez.png', cv2.IMREAD_COLOR)
if image is None:
    raise FileNotFoundError("Image not found!")

# Convert image from BGR to RGB for display in matplotlib
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

height, width, channels = image.shape

# Define Gaussian noise parameters
mu = 0
sigma = random.uniform(2, 10)  # You can use a fixed value here for testing if needed

# Save mu and sigma to CSV
file_path = './SkyteParametreSumuCSV/random_mu_sigma.csv'
file_exists = os.path.isfile(file_path)
with open(file_path, mode='a', newline='') as file:
    writer = csv.writer(file)
    if not file_exists:
        writer.writerow(['mu', 'sigma'])
    writer.writerow([mu, sigma])

# Add Gaussian noise to each channel
noisy_image = np.empty_like(image)

for c in range(channels):  # Loop over B, G, R channels
    noise = np.random.normal(mu, sigma, size=(height, width)).astype(np.int16)
    noisy_channel = image[:, :, c].astype(np.int16) + noise
    noisy_channel = np.clip(noisy_channel, 0, 255).astype(np.uint8)
    noisy_image[:, :, c] = noisy_channel

# Save noisy image
for index in range(0, 100):
    filename = f'ZasumenyRealnyObrazokPozityVHistograme_{index}.png'
    if not os.path.isfile(filename):
        cv2.imwrite(filename, noisy_image)
        break

# Convert to RGB for display
noisy_image_rgb = cv2.cvtColor(noisy_image, cv2.COLOR_BGR2RGB)

# Plot original vs noisy image
plt.figure(figsize=(10, 5))

plt.subplot(1, 2, 1)
plt.title("Obrázok s Gaussovským šumom")
plt.imshow(noisy_image_rgb)
plt.axis('off')

plt.subplot(1, 2, 2)
plt.title("Histogram (všetky kanály)")
colors = ['r', 'g', 'b']
for i, color in enumerate(colors):
    plt.hist(noisy_image_rgb[:, :, i].ravel(), bins=256, range=(0, 256), color=color, alpha=0.5, label=f'{color.upper()}')
plt.xlabel("Intenzita pixelov")
plt.ylabel("Počet pixelov", fontsize=9)
plt.legend()

# Save the Histogram
output_dir = "Histogramy"
os.makedirs(output_dir, exist_ok=True)
plt.savefig(os.path.join(output_dir, "HistogramZasumenehoRealmehoObrazku_Color.png"), dpi=300)

plt.tight_layout()
plt.show()
