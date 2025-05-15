import cv2
import numpy as np

# Load ObrazkyJPEG (color)
image = cv2.imread('../ObrazkyJPEG/Cat.jpg')

# Generate Gaussian noise
mean = 0
stddev = 25  # Adjust this for more/less noise
gaussian_noise = np.random.normal(mean, stddev, image.shape).astype(np.float32)

# Add noise to ObrazkyJPEG
noisy_image = image.astype(np.float32) + gaussian_noise

# Clip values to [0, 255] and convert back to uint8
noisy_image = np.clip(noisy_image, 0, 255).astype(np.uint8)

# Save the result
cv2.imwrite('noisy_image.jpg', noisy_image)

print("Noisy ObrazkyJPEG saved as 'noisy_image.jpg'")
