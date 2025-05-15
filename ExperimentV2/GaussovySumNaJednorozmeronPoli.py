#!/usr/bin/python3

import numpy as np

# Funkcia generujúca Gaussov šum pomocou Box-Mullerovej transformácie
def generate_gaussian_noise(mean, std_dev, size):
    u1 = np.random.rand(size)
    u2 = np.random.rand(size)
    z0 = np.sqrt(-2 * np.log(u1)) * np.cos(2 * np.pi * u2)
    return z0 * std_dev + mean

# Funkcia generujúca jednorozmerné Gaussovské jadro (filter)
def gaussian_1d_kernel(ksize, sigma):
    assert ksize % 2 == 1, "Veľkosť jadra musí byť nepárna"
    half = ksize // 2
    x = np.arange(-half, half + 1)
    kernel = np.exp(-x**2 / (2 * sigma**2))
    kernel /= kernel.sum()  # Normalizácia jadra
    return kernel

# Vytvorenie pôvodného poľa (napr. jasová hodnota 130)
arr = np.full(20, 130)
print("\nPôvodné pole (bez šumu):")
print(arr)

# Generovanie Gaussovho šumu a aplikovanie na pôvodné pole
noise = generate_gaussian_noise(mean=0, std_dev=5, size=arr.size)
noisy_arr = arr + noise
print("\nPole so šumom:")
print(noisy_arr)

# Funkcia na filtrovanie šumu pomocou 1D Gaussovského filtra
def filtruj(noisy_arr):
    mean_value = np.mean(noisy_arr)
    std_dev_value = np.std(noisy_arr)

    if std_dev_value == 0:
        return noisy_arr

    print("\n--- Parametre pred filtráciou ---")
    print(f"Stredná hodnota (mean): {mean_value}")
    print(f"Štandardná odchýlka (std dev): {std_dev_value}")

    sigma = std_dev_value
    ksize = int(6 * sigma + 1)
    if ksize % 2 == 0:
        ksize += 1
    kernel = gaussian_1d_kernel(ksize, sigma)

    print(f"Veľkosť jadra (ksize): {ksize}")
    kradius = ksize // 2
    print(f"Polomer jadra (kradius): {kradius}")
    print(f"Gaussovské jadro (kernel): {kernel}")

    smoothed_arr = np.zeros((arr.size))

    for i in range(arr.size):
        sucet = 0
        for j in range(-kradius, kradius + 1):
            k = i + j
            if k >= arr.size:
                k = k - arr.size
            sucet += kernel[j + kradius] * noisy_arr[k]
        smoothed_arr[i] = min(255, max(0, sucet))

    print("\nVýsledok po filtrovaní (vyhladené pole):")
    print(smoothed_arr)
    return smoothed_arr

# Trojnásobná iterácia filtrovania
for i in range(3):
    print(f"\n\n=============== Iterácia {i + 1} ===============")
    noisy_arr = filtruj(noisy_arr)
