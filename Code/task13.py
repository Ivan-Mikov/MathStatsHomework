from scipy.stats import f
import matplotlib.pyplot as plt
import numpy as np

# Параметры распределения
n = 139   # dfn = n-1
s1_length = 5.722
s1_width = 4.612

m = 1000  # dfd = m-1
s2_length = 6.161
s2_width = 5.055

x_length = s1_length ** 2 / s2_length ** 2
x_width = s1_width ** 2 / s2_width ** 2

alpha = 0.05

# Вычисление интеграла от 0 до x
F_left = f.ppf(alpha / 2, n-1, m-1)
F_right = f.ppf(1 - alpha / 2, n-1, m-1)

print(x_length)
print(x_width)

print(F_left, F_right)


def power(tetha):
    return f.cdf(0.77 / tetha, n-1, m-1) + 1 - (f.cdf(1.27 / tetha, n-1, m-1))


theta_values = np.linspace(0.1, 3, 500)

# Вычисление мощности
power_values = [power(theta) for theta in theta_values]

# Построение графика
plt.figure(figsize=(10, 6))
plt.plot(theta_values, power_values, 'b-', linewidth=2, label=f'F({n-1}, {m-1})')

# Настройка графика
plt.xlabel('θ (отношение дисперсий)', fontsize=12)
plt.ylabel('Мощность критерия', fontsize=12)
plt.title(f'Функция мощности для F-распределения (n={n}, m={m})', fontsize=14)
plt.grid(True, alpha=0.3)
plt.legend()

# Отметим характерные точки
plt.axhline(y=0.05, color='r', linestyle='--', alpha=0.5, label='Уровень значимости 0.05')
plt.axvline(x=1.0, color='g', linestyle='--', alpha=0.5, label='θ = 1 (нулевая гипотеза)')

plt.legend()
plt.tight_layout()
plt.savefig("./T13.png")
