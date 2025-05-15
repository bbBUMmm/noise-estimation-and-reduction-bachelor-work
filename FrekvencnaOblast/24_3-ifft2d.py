#!/usr/bin/python3

"""24_3-ifft2d.py: dvojrozmerna FFT, uprava vo frekvencnej domene, inverzna FFT pomocou implementacie z kniznice Numpy"""
__author__ = "Michal Vagac"
__email__ = "michal.vagac@gmail.com"

from PIL import Image
import math
import numpy as np
import pylab as py
import matplotlib.pyplot as plt

# nacitaj obrazok a skonvertuj ho do odtienov sedej
fxy = Image.open('data/balon.jpg').convert('L')
sirka, vyska = fxy.size

# pocitaj FT
Fuv = np.fft.fft2(fxy)

# definuj filter
D0 = 30
# 1) lowpass filter
Huv = np.zeros((vyska, sirka))
Huv[int(vyska / 2 - D0):int(vyska / 2 + D0), int(sirka / 2 - D0):int(sirka / 2 + D0)] = 1
# 2) highpass filter
# Huv = np.ones((vyska, sirka))
# Huv[int(vyska/2-D0):int(vyska/2+D0), int(sirka/2-D0):int(sirka/2+D0)] = 0

# uprav data frekvencnej domeny
Guv = Huv * np.fft.fftshift(Fuv)

# pocitaj IFT
hxy = np.abs(np.fft.ifft2(np.fft.ifftshift(Guv)))

vysledok = Image.fromarray(np.asarray(hxy).astype(np.uint8), 'L')
vysledok.save('vysledok.png')

Fuv[0, 0] = 0  # odstran DC koli vizualizacii
PSDF = np.abs(Fuv) ** 2  # 2D power spectrum
Guv[0, 0] = 0  # odstran DC koli vizualizacii
PSDG = np.abs(Guv) ** 2  # 2D power spectrum

# zobraz
fig = plt.figure()
ax1 = fig.add_subplot(221)
ax1.imshow(fxy, cmap='gray')
ax1.set_xlabel('priestorova domena')
ax2 = fig.add_subplot(223)
ax2.imshow(np.log10(np.fft.fftshift(PSDF)), cmap='gray')
ax2.set_xlabel('frekvencna domena (log10)')
ax3 = fig.add_subplot(222)
ax3.imshow(hxy, cmap='gray')
ax3.set_xlabel('priestorova domena')
ax4 = fig.add_subplot(224)
ax4.imshow(np.log10(PSDG), cmap='gray')
ax4.set_xlabel('frekvencna domena (log10)')
plt.show()
#fig.savefig('vysledok.png')

