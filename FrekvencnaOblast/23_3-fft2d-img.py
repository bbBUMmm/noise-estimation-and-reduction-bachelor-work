#!/usr/bin/python3

"""23_3-fft2d-img.py: dvojrozmerna fourierova transformacia nacitanych obrazkov pomocou implementacie z kniznice Numpy"""
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
ds = 20
dv = 20

# pocitaj FT
Fuv = np.fft.fft2(fxy)
Fuv[0, 0] = 0  # odstran DC koli vizualizacii
PSD = np.abs(Fuv) ** 2  # 2D power spectrum
# amplitudove_spektrum = 20 * np.log(np.abs(fshift))

# zobraz
fig = plt.figure()
ax1 = fig.add_subplot(221)
ax1.imshow(fxy, cmap='gray')
ax1.set_xlabel('priestorova domena')
ax2 = fig.add_subplot(222)
ax2.imshow(np.fft.fftshift(PSD), cmap='gray')
ax2.set_xlabel('frekvencna domena')
ax3 = fig.add_subplot(223)
ax3.imshow(np.log10(np.fft.fftshift(PSD)), cmap='gray')
ax3.set_xlabel('frekvencna domena (log10)')
ax4 = fig.add_subplot(224)
ax4.imshow(np.fft.fftshift(PSD)[int(vyska / 2 - dv):int(vyska / 2 + dv), int(sirka / 2 - ds):int(sirka / 2 + ds)], cmap='gray')
ax4.set_xlabel('frekvencna domena (detail)')
plt.show()
fig.savefig('vysledok.png')

# obrazok = cv2.imread('balon.jpg', cv2.IMREAD_GRAYSCALE)
# f = np.fft.fft2(obrazok)
# fshift = np.fft.fftshift(f)
# vyska, sirka = obrazok.shape
# vyska_stred, sirka_stred = vyska/2, sirka/2
# fshift[vyska_stred - 30:vyska_stred + 30, sirka_stred - 30:sirka_stred + 30] = 0		# filter
# f_ishift = np.fft.ifftshift(fshift)
# vysledok = np.fft.ifft2(f_ishift)
# vysledok = np.abs(vysledok)
# plt.subplot(131), plt.imshow(obrazok, cmap = 'gray')
# plt.title('vstup'), plt.xticks([]), plt.yticks([])
# plt.subplot(132), plt.imshow(vysledok, cmap = 'gray')
# plt.title('vysledok po HPF'), plt.xticks([]), plt.yticks([])
# plt.subplot(133), plt.imshow(vysledok)
# plt.title('vysledok v JET'), plt.xticks([]), plt.yticks([])
# plt.show()
