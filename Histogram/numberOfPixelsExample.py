import numpy as np
import matplotlib.pyplot as plt

# Create a grayscale ObrazkyJPEG with a background brightness of 50 (0-255)
image = np.full((256, 256), 50, dtype=np.uint8)

# Plot the ObrazkyJPEG and its Histogram
plt.figure(figsize=(10, 4))

plt.subplot(1, 2, 1)
plt.title("Ideálny obrázok bez šumu")
plt.imshow(image, cmap='gray')
plt.axis('off')

plt.subplot(1, 2, 2)
plt.title("Histogram")
plt.hist(image.ravel(), bins=256, range=(0, 256), color='gray')
plt.xlabel("Hodnoty intenzity pixelov")
plt.ylabel("Počet pixlov tejto hodnoty")

plt.tight_layout()
plt.show()
