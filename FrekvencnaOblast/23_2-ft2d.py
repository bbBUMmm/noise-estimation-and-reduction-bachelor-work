#!/usr/bin/python3

"""23_2-ft2d.py: dvojrozmerna fourierova transformacia"""
__author__ = "Michal Vagac"
__email__ = "michal.vagac@gmail.com"

import time
import cmath
import math
import matplotlib.pyplot as plt

from PIL import Image


# TODO dako to nefunguje

def DFT_manual(fxy, centruj=False):
    M, N = fxy.size
    fxy2 = fxy.copy()
    if centruj:
        for y in range(0, N):
            for x in range(0, M):
                fxy2.putpixel((x, y), math.pow(-1, x + y) * fxy.getpixel((x, y)))
    Fuv = [[0 for x in range(M)] for y in range(N)]
    mymax = 0
    for v in range(0, N):
        print(v, ' / ', N)
        for u in range(0, M):
            sum = 0
            for y in range(0, N):
                for x in range(0, M):
                    sum += fxy2.getpixel((x, y)) * cmath.exp(-2j * cmath.pi * (u * x / M + v * y / N))
            # pocitaj magnitudu (amplitudove spektrum)
            val = abs(sum)
            if val > mymax:
                mymax = val
            Fuv[u][v] = val
    # preskaluj na odtiene sedej
    r = 255.0 / mymax
    for v in range(0, N):
        for u in range(0, M):
            Fuv[u][v] = Fuv[u][v] * r

    print(Fuv)
    FuvObr = Image.new('L', (M, N))
    FuvObr.putdata(Fuv)
    return FuvObr


# vytvor obrazok odtienov sedej
fxy = Image.new('L', (50, 50))
sirka, vyska = fxy.size

# priprav (generuj) data #1: 1 schod
# for y in range(0, vyska):
#	for x in range(0, sirka):
#		if x < 20:
#			fxy.putpixel((x, y), 255)
#		else:
#			fxy.putpixel((x, y), 0)
# priprav (generuj) data #2: sinus
for y in range(0, vyska):
    for x in range(0, sirka):
        j = float(x) / sirka * 2 * math.pi
        fxy.putpixel((x, y), vyska / 2 + vyska / 2 * math.sin(3.0 * j))
# priprav (generuj) data #3: hlavna frekvencia, na ktorej je namodulovana druha frekvencia
# for y in range(0, vyska):
#	for x in range(0, sirka):
#		j = float(x)/sirka*2*math.pi
#		fxy.putpixel((x, y), 127 + 110*(10*math.cos(2.0*j)+math.sin(50.0*j)))

# pocitaj DFT
l = time.time()
Fuv = DFT_manual(fxy, False)
print("--- %s sekund ---" % (time.time() - l))

# zobraz
fig = plt.figure()
ax1 = fig.add_subplot(121)
ax1.imshow(fxy, cmap='gray')
ax1.set_xlabel('priestorova domena')
ax2 = fig.add_subplot(122)
ax2.imshow(Fuv, cmap='gray')
ax2.set_xlabel('frekvencna domena')
plt.show()
fig.savefig('vysledok.png')
