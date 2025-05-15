#!/usr/bin/python3

# https://jakevdp.github.io/blog/2013/08/28/understanding-the-fft/
# https://numericalrecipes.wordpress.com/2009/05/29/the-cooley-tukey-fft-algorithm-for-general-factorizations/
# http://www.drdobbs.com/cpp/a-simple-and-efficient-fft-implementatio/199500857

"""23_1-ft1d-np.py: jednorozmerna fourierova transformacia pomocou kniznice Numpy"""
__author__ = "Michal Vagac"
__email__ = "michal.vagac@gmail.com"

import time
import numpy as np
import matplotlib.pyplot as plt


def DFT_np(fx):
    """Compute the discrete Fourier Transform of the 1D array x"""
    fx = np.asarray(fx, dtype=float)
    N = fx.shape[0]
    u = np.arange(N)
    x = u.reshape((N, 1))
    M = np.exp(-2j * np.pi * x * u / N)
    return np.dot(M, fx)


def FFT_np(x):
    """A recursive implementation of the 1D Cooley-Tukey FFT"""
    x = np.asarray(x, dtype=float)
    N = x.shape[0]

    if N % 2 > 0:
        raise ValueError("size of x must be a power of 2")
    elif N <= 32:  # this cutoff should be optimized
        return DFT_np(x)
    else:
        X_even = FFT_np(x[::2])
        X_odd = FFT_np(x[1::2])
        factor = np.exp(-2j * np.pi * np.arange(N) / N)
        return np.concatenate([X_even + factor[:N / 2] * X_odd, X_even + factor[N / 2:] * X_odd])


# priprav (generuj) data #1: nahodne
# fx = np.random.random(1024)
# priprav (generuj) data #2: hlavna frekvencia, na ktorej je namodulovana druha frekvencia
x = np.linspace(0.0, 512, 512)
xpi = 2.0 * np.pi * x
fx = 10 * np.cos(2.0 * xpi) + np.sin(50.0 * xpi)

# pocitaj DFT
l = time.time()
# print(np.allclose(DFT_np(fx), np.fft.fft(fx)))
Fu = DFT_np(fx)
print("--- %s sekund ---" % (time.time() - l))

# zobraz
fig = plt.figure()
ax1 = fig.add_subplot(121)
ax1.plot(fx)
ax1.set_xlabel('casova domena')
ax2 = fig.add_subplot(122)
ax2.plot(range(int(-len(Fu) / 2), int(len(Fu) / 2)), Fu)
ax2.set_xlabel('frekvencna domena')
plt.show()
fig.savefig('vysledok.png')
