#!/usr/bin/python3

"""25_3-ft_filter.py: dvojrozmerna FFT, uprava vo frekvencnej domene, inverzna FFT pomocou implementacie z kniznice Numpy; zobrazenie pouziteho filtra"""
__author__ = "Michal Vagac"
__email__ = "michal.vagac@gmail.com"

from PIL import Image
import math
import numpy as np
import pylab as py
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# nacitaj obrazok a skonvertuj ho do odtienov sedej
fxy = Image.open('data/balon.jpg').convert('L')
sirka, vyska = fxy.size

# pocitaj FT
Fuv = np.fft.fft2(fxy)

# definuj filter
D0 = 50
W = 23
n = 3


def D(u, v):
    return math.sqrt((u - sirka / 2) ** 2 + (v - vyska / 2) ** 2)


Huv = np.zeros((vyska, sirka))
for v in range(0, vyska):
    for u in range(0, sirka):
        # 1) idealny
        #		if D(u, v) <= D0:
        #			Huv[v, u] = 1
        #		else:
        #			Huv[v, u] = 0
        # 2) Butterworthov dolnopriepustny
        # Huv[v, u] = 1 / (1 + (D(u, v) / D0) ** (2 * n))
        # 3) Gausov dolnopriepustny
        Huv[v, u] = np.exp(-D(u, v)**2 / ( 2 * D0 * D0 ))
    # 3) Butterworthov typu pasmovej zadrze
# Huv[v, u] = 1 / (1 + ( (D(u, v)*W) / (D(u, v)**2-D0**2+0.0000001) )**(2*n))

# Huv = np.ones((vyska, sirka)) - Huv

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
ax1 = fig.add_subplot(231)
ax1.imshow(fxy, cmap='gray')
ax1.set_xlabel('priestorova domena')
ax2 = fig.add_subplot(234)
ax2.imshow(np.log10(np.fft.fftshift(PSDF)), cmap='gray')
ax2.set_xlabel('frekvencna domena (log10)')
ax3 = fig.add_subplot(233)
ax3.imshow(hxy, cmap='gray')
ax3.set_xlabel('priestorova domena')
ax4 = fig.add_subplot(236)
ax4.imshow(np.log10(PSDG), cmap='gray')
ax4.set_xlabel('frekvencna domena (log10)')
ax5 = fig.add_subplot(235, projection='3d')
x, y = np.mgrid[:vyska, :sirka]
ax5.plot_surface(x, y, Huv, rstride=30, cstride=30, cmap=plt.cm.jet, linewidth=0., antialiased=False)
# ax5.plot_wireframe(x, y, Huv, rstride=20, cstride=20, cmap=plt.cm.gray, linewidth=1, antialiased=False)
ax5.set_xlabel('filter')
plt.show()
#fig.savefig('vysledok.png')

