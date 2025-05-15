#!/usr/bin/python3

"""24_1-ifft1d.py: jednorozmerna FFT, uprava vo frekvencnej domene, inverzna FFT pomocou implementacie z kniznice Numpy"""
__author__ = "Michal Vagac"
__email__ = "michal.vagac@gmail.com"

import numpy as np
import matplotlib.pyplot as plt

# priprav (generuj) data: hlavna frekvencia, na ktorej je namodulovana druha frekvencia
n = 300
x = np.linspace(0.0, n, n)
xpi = 2.0 * np.pi * x
fx = 10 * np.cos(2.0 * xpi) + np.sin(50.0 * xpi)

# pocitaj FT
Fu = np.fft.fft(fx)

# definuj filter
D0 = 30
# 1) lowpass filter
Hu = np.linspace(0, 0, n)
Hu[int(n / 2 - D0):int(n / 2 + D0)] = 1
# 2) highpass filter
# Hu = np.linspace(1, 1, n)
# Hu[int(n/2-D0):int(n/2+D0)] = 0

# uprav data frekvencnej domeny
Gu = Hu * np.fft.fftshift(Fu)

# pocitaj IFT
hx = np.fft.ifft(np.fft.ifftshift(Gu))

# zobraz
fig = plt.figure()
ax1 = fig.add_subplot(141)
ax1.plot(fx)
ax1.set_xlabel('casova domena')
ax2 = fig.add_subplot(142)
ax2.plot(range(int(-len(Fu) / 2), int(len(Fu) / 2)), np.fft.fftshift(np.abs(Fu)))
# ax2.plot(range(int(-len(Fu)/2), int(len(Fu)/2)), np.abs(Fu))
ax2.set_xlabel('frekvencna domena')
ax3 = fig.add_subplot(143)
ax3.set_ylim((0, 1600))
ax3.plot(range(int(-len(Gu) / 2), int(len(Gu) / 2)), np.abs(Gu))
ax3.set_xlabel('frekvencna domena po zmene')
ax4 = fig.add_subplot(144)
ax4.set_ylim((-15, 15))
ax4.plot(hx)
ax4.set_xlabel('casova domena po IFT')
plt.show()
fig.savefig('vysledok.png')
