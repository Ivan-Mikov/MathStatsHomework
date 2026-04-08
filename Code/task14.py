from scipy.stats import norm
import numpy as np
import matplotlib.pyplot as plt

# Параметры
alpha = 0.05   # например, 5%

# Квантиль порядка 1 - alpha
quantile = norm.ppf(1 - alpha)

print(f"Квантиль порядка {1-alpha} для N(0, 1): {quantile:.4f}")

x = np.array([-1.11, -6.1, 2.42])
y = np.array([-2.29, -2.91])

x_ = np.mean(x)
y_ = np.mean(y)

delta = (x_ - y_) / np.sqrt(7/6)

print(x_, y_, delta)


def power(tetha):
    return 1 - norm.cdf(1.645 - tetha / np.sqrt(7/6))


theta_values = np.linspace(0, 10, 500)

# Вычисление мощности
power_values = [power(theta) for theta in theta_values]

# Построение графика
plt.figure(figsize=(10, 6))
plt.plot(theta_values, power_values, 'b-', linewidth=2)

# Настройка графика
plt.xlabel('θ (a - b)', fontsize=12)
plt.ylabel('Мощность критерия', fontsize=12)
plt.grid(True, alpha=0.3)

# Отметим характерные точки
plt.axhline(y=0.05, color='r', linestyle='--', alpha=0.5, label='Уровень значимости 0.05')
plt.axvline(x=0, color='g', linestyle='--', alpha=0.5, label='θ = 0 (нулевая гипотеза)')

plt.legend()
plt.tight_layout()
plt.savefig("./T14.png")
