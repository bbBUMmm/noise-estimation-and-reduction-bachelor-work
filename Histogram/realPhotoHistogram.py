import cv2
import matplotlib.pyplot as plt

# Nacitanie obrazku
image = cv2.imread('../Obrazky/ObrazkyJPEG/VyrezyAleboSegmentyObrazkuJPEG/zelenaZnackaVyrez.jpg')
# image = cv2.imread('../ObrazkyJPEG/modraZnacka.jpg')
# ObrazkyJPEG = cv2.imread('../ObrazkyJPEG/budova.jpg')
# ObrazkyJPEG = cv2.imread('../ObrazkyJPEG/zelenaZnacka.jpg')
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB for matplotlib

# Plot
plt.figure(figsize=(10, 4))

plt.subplot(1, 2, 1)
plt.title("Originalný obrázok")
plt.imshow(image_rgb)
plt.axis('off')

# Plot histograms
plt.subplot(1, 2, 2)
colors = ('r', 'g', 'b')
for i, col in enumerate(colors):
    hist = cv2.calcHist([image], [i], None, [256], [0, 256])
    plt.plot(hist, color=col)
    plt.xlim([0, 256])

plt.title("Histogram")
plt.xlabel("Hodnoty intenzity pixelov")
plt.ylabel("Počet pixlov tejto hodnoty")

plt.tight_layout()
plt.show()
