import cv2
import matplotlib.pyplot as plt


# Load in grayscale mode
# ObrazkyJPEG = cv2.imread('../ObrazkyJPEG/zelenaZnacka.jpg', cv2.IMREAD_GRAYSCALE)
# image = cv2.imread('../ObrazkyJPEG/modraZnacka.jpg', cv2.IMREAD_GRAYSCALE)
# ObrazkyJPEG = cv2.imread('../ObrazkyJPEG/budova.jpg', cv2.IMREAD_GRAYSCALE)
image = cv2.imread('../Obrazky/ObrazkyJPEG/VyrezyAleboSegmentyObrazkuJPEG/zelenaZnackaVyrez.jpg', cv2.IMREAD_GRAYSCALE)

# Show
plt.figure(figsize=(10, 4))

plt.subplot(1, 2, 1)
plt.title("Grayscale Obrázok")
plt.imshow(image, cmap='gray')
plt.axis('off')

# Show the Histogram
plt.subplot(1, 2, 2)
plt.title("Grayscale Histogram")
plt.hist(image.ravel(), bins=256, range=(0, 256), color='gray')
plt.xlabel("Hodnoty intenzity pixelov")
plt.ylabel("Počet pixlov tejto hodnoty")


plt.tight_layout()
plt.show()
