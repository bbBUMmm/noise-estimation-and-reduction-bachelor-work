#!/usr/bin/python3

"""23_3-fft2d-f.py: dvojrozmerna fourierova transformacia generovanych obrazkov pomocou implementacie z kniznice Numpy"""
__author__ = "Michal Vagac"
__email__ = "michal.vagac@gmail.com"

from PIL import Image
import math
import numpy as np
# import pylab as py
import matplotlib.pyplot as plt

# vytvor obrazok odtienov sedej
fxy = Image.new('L', (100, 100))
sirka, vyska = fxy.size

# priprav (generuj) data #1: 1 schod
# for y in range(0, vyska):
#	for x in range(0, sirka):
#		if x < 20:
#			fxy.putpixel((x, y), 255)
#		else:
#			fxy.putpixel((x, y), 0)
# priprav (generuj) data #2: sinus
maxl = math.sqrt(sirka ** 2 + vyska ** 2)
maxl = sirka + vyska
for y in range(0, vyska):
    for x in range(0, sirka):
        j = float(x) / sirka * 2 * math.pi
        j2 = float(x) / vyska * 2 * math.pi
        j = float(math.sqrt(x ** 2 + y ** 2)) / maxl * 2 * math.pi
        j = float(x + y) / maxl * 2 * math.pi
        fxy.putpixel((x, y), int(127 + 64 * (math.sin(10.0 * j) + math.sin(3 * j))))
        fxy.putpixel((x, y), int(127 + 127 * math.sin(3 * j)))
# priprav (generuj) data #3: hlavna frekvencia, na ktorej je namodulovana druha frekvencia
# for y in range(0, vyska):
#	for x in range(0, sirka):
#		j = float(x)/sirka*2*math.pi
#		fxy.putpixel((x, y), 127 + 110*(10*math.cos(2.0*j)+math.sin(50.0*j)))

# pocitaj FT
Fuv = np.fft.fft2(fxy)
Fuv[0, 0] = 0  # odstran DC koli vizualizacii
PSD = np.abs(Fuv) ** 2  # 2D power spectrum

# zobraz
fig = plt.figure()
ax1 = fig.add_subplot(121)
ax1.imshow(fxy, cmap='gray')
ax1.set_xlabel('priestorova domena')
ax2 = fig.add_subplot(122)
ax2.imshow(np.fft.fftshift(PSD), cmap='gray')
ax2.set_xlabel('frekvencna domena')
plt.show()
fig.savefig('vysledok.png')
