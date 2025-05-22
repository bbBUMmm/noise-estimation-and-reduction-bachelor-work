# Skript na generaciu obrazku s ukazkovym histogramom

import numpy as np
import matplotlib.pyplot as plt

# Rewrote function formula for the Gaussian random variable distribution
def gaussian_probability(x, mu, sigma):
    # 1. Coefficient part: 1 / (σ * sqrt(2π))
    coefficient = 1 / (sigma * np.sqrt(2 * np.pi))

    # 2. Exponent part: e^(-(x - μ)^2 / (2σ^2))
    exponent = np.exp(-((x - mu) ** 2) / (2 * sigma ** 2))

    # 3. Full formula
    return coefficient * exponent

# Value x
x = np.linspace(0, 255, 500)

# Два набори параметрів
mu1, sigma1 = 128, 20
mu2, sigma2 = 100, 40

# Вираховуємо значення функції
y1 = gaussian_probability(x, mu1, sigma1)
y2 = gaussian_probability(x, mu2, sigma2)

# Побудова графіків
plt.figure(figsize=(10, 6))
plt.plot(x, y1, label='μ = 128, σ = 20', color='blue')
plt.plot(x, y2, label='μ = 100, σ = 40', color='orange')
plt.title('Gaussove rozdelenia s rôznymi parametrami')
plt.xlabel('x (hodnota intenzity pixelu x)')
plt.ylabel('Hustota pravdepodobnosti')
plt.legend()
plt.grid(True)
plt.show()
