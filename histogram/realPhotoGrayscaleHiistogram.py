import cv2
import matplotlib.pyplot as plt


# Load the ObrazkyJPEG in grayscale mode
# ObrazkyJPEG = cv2.imread('../ObrazkyJPEG/greenSign.jpg', cv2.IMREAD_GRAYSCALE)
# image = cv2.imread('../ObrazkyJPEG/blueSign.jpg', cv2.IMREAD_GRAYSCALE)
# ObrazkyJPEG = cv2.imread('../ObrazkyJPEG/building.jpg', cv2.IMREAD_GRAYSCALE)
image = cv2.imread('../ObrazkyJPEG/VyrezyAleboSegmentyObrazkuJPEG/greenSignCut.jpg', cv2.IMREAD_GRAYSCALE)

# Show the ObrazkyJPEG
plt.figure(figsize=(10, 4))

plt.subplot(1, 2, 1)
plt.title("Grayscale Obrázok")
plt.imshow(image, cmap='gray')
plt.axis('off')

# Show the histogram
plt.subplot(1, 2, 2)
plt.title("Grayscale Histogram")
plt.hist(image.ravel(), bins=256, range=(0, 256), color='gray')
plt.xlabel("Hodnoty intenzity pixelov")
plt.ylabel("Počet pixlov tejto hodnoty")


plt.tight_layout()
plt.show()
