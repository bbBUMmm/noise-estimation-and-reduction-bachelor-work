import numpy as np
import cv2
import matplotlib.pyplot as plt

# Create a grayscale ObrazkyJPEG with a background brightness of 50 (0-255)
image = np.full((256, 256), 0, dtype=np.uint8)

# Draw a medium gray square on the original ObrazkyJPEG
square_image = image.copy()
top_left = (64, 64)
bottom_right = (192, 192)
square_image = cv2.rectangle(square_image, top_left, bottom_right, color=150, thickness=-1)

# Draw a light gray circle on the original ObrazkyJPEG
circle_image = image.copy()
center = (128, 128)
radius = 50
circle_image = cv2.circle(circle_image, center, radius, color=200, thickness=-1)  # Light gray circle

# Draw a darker gray triangle on the original ObrazkyJPEG
triangle_image = image.copy()
triangle_points = np.array([
    [128, 100],  # top
    [100, 150],  # bottom left
    [156, 150]   # bottom right
], np.int32)
triangle_image = cv2.fillPoly(triangle_image, [triangle_points], color=255)  # Lighter gray triangle

# Combine the shapes into one ObrazkyJPEG
final_image = np.maximum(np.maximum(square_image, circle_image), triangle_image)

# Plot the final ObrazkyJPEG and its Histogram
plt.figure(figsize=(10, 4))

plt.subplot(1, 2, 1)
plt.title("Ideálny obrázok bez šumu")
plt.imshow(final_image, cmap='gray')
plt.axis('off')

plt.subplot(1, 2, 2)
plt.title("Histogram")
plt.hist(final_image.ravel(), bins=256, range=(0, 256), color='gray')
plt.xlabel("Hodnoty intenzity pixelov")
plt.ylabel("Počet pixlov tejto hodnoty")

plt.tight_layout()
plt.show()
