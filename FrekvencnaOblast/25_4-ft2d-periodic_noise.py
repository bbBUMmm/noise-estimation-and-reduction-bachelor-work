#!/usr/bin/python3

"""25_4-ft2d-periodic_noise.py: dvojrozmerna FFT, peridicky filter"""
__author__ = "Michal Vagac"
__email__ = "michal.vagac@gmail.com"

import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

fxy = Image.open('data/balon.jpg').convert('L')
sirka, vyska = fxy.size
fxy = np.array(fxy)
Fuv = np.fft.fft2(fxy)

Huv = np.ones((vyska, sirka))
Guv = np.fft.ifftshift(Huv * np.fft.fftshift(Fuv))

# poskodenie
q = 1000000
d = 20
Guv[d][d] = q
Guv[-d][d] = q
Guv[d][-d] = q
Guv[-d][-d] = q
gxy = np.abs(np.fft.ifft2(Guv))

# obnova
Guv2 = np.fft.fft2(gxy)
Guv2[d][d] = 1
Guv2[-d][d] = 1
Guv2[d][-d] = 1
Guv2[-d][-d] = 1
gxy2 = np.abs(np.fft.ifft2(Guv2))

fig = plt.figure()
ax1 = fig.add_subplot(231)
ax1.imshow(fxy, cmap='gray')
ax2 = fig.add_subplot(232)
ax2.imshow(gxy, cmap='gray')
ax3 = fig.add_subplot(233)
ax3.imshow(gxy2, cmap='gray')

ax4 = fig.add_subplot(234)
ax4.imshow(np.log10(np.abs(np.fft.fftshift(Fuv))**2), cmap='gray')
ax5 = fig.add_subplot(235)
ax5.imshow(np.log10(np.abs(np.fft.fftshift(Guv))**2), cmap='gray')
ax6 = fig.add_subplot(236)
ax6.imshow(np.log10(np.abs(np.fft.fftshift(Guv2))**2), cmap='gray')
plt.show()
