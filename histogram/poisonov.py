import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

# Load and convert to grayscale
image = Image.open("LennaTestImage.png").convert("L")
image_array = np.array(image)

# Normalize to [0, 1]
image_normalized = image_array / 255.0

# Apply Poisson noise
poisson_noisy = np.random.poisson(image_normalized * 255) / 255.0
poisson_noisy = np.clip(poisson_noisy, 0, 1)
poisson_noisy_image = (poisson_noisy * 255).astype(np.uint8)

# Display only the noisy image, no text or axes
plt.imshow(poisson_noisy_image, cmap='gray')
plt.axis('off')
plt.show()
