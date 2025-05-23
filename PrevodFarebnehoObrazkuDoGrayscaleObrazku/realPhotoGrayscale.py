import cv2
import matplotlib.pyplot as plt

# Nacitat obrazok
original_image = cv2.imread('../Obrazky/ObrazkyJPEG/VyrezyAleboSegmentyObrazkuJPEG/zelenaZnackaVyrez.jpg')
image_gray = cv2.imread('../Obrazky/ObrazkyJPEG/VyrezyAleboSegmentyObrazkuJPEG/zelenaZnackaVyrez.jpg', cv2.IMREAD_GRAYSCALE)

# OpenCV nacita obrazok v BGR by default ale mplib ocakava rhb
original_image_rgb = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)

# Plot
plt.figure(figsize=(10, 4))

plt.subplot(1, 2, 1)
plt.title("Farebný Obrázok")
plt.imshow(original_image_rgb)
plt.axis('off')

plt.subplot(1, 2, 2)
plt.title("Grayscale Obrázok")
plt.imshow(image_gray, cmap='gray')
plt.axis('off')

plt.tight_layout()
plt.show()
