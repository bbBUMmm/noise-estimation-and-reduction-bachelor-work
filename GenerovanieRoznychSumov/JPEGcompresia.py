import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import io

# Načítanie pôvodného obrázka
povodny_obrazok = Image.open("Inputs/LennaTestImage.png").convert("RGB")

# Aplikovanie JPEG kompresie s nízkou kvalitou
buffer = io.BytesIO()
povodny_obrazok.save(buffer, format="JPEG", quality=10)
buffer.seek(0)

# Načítanie komprimovaného obrázka
komprimovany_obrazok = Image.open(buffer)

# Konverzia obrázkov na polia
povodny_array = np.array(povodny_obrazok)
komprimovany_array = np.array(komprimovany_obrazok)

# Zobrazenie porovnania
plt.figure(figsize=(14, 7))  # väčšia veľkosť pre viac miesta

plt.subplot(1, 2, 1)
plt.imshow(povodny_array)
plt.axis('off')
plt.title('Pôvodný obrázok', fontsize=16, pad=20)  # väčší padding

plt.subplot(1, 2, 2)
plt.imshow(komprimovany_array)
plt.axis('off')
plt.title('Obrázok po JPEG kompresii (kvalita = 10)', fontsize=16, pad=20)

# Lepšie prispôsobenie rozloženia
plt.subplots_adjust(top=0.85, bottom=0.05, left=0.05, right=0.95, wspace=0.15)

plt.show()
