# Здесь представлен код, с помощью которого я решал задачу №2.
# Выборка генерируется без сида, поэтому при повторном запуске
# числовые характеристики, графики и оценки могут отличаться.

import math
import numpy as np
import matplotlib.pyplot as plt

def asym_coeff(x): # Вычисляет коэффициент ассиметрии для выборки x
    central_moment_2 = np.mean((x - np.mean(x))**2)
    central_moment_3 = np.mean((x - np.mean(x))**3)
    return central_moment_3 / (central_moment_2 ** 1.5)

def F(x):
    result = np.zeros_like(x)
    mask = x >= 0
    result[mask] = 1 - np.exp(-x[mask])
    return result

def med_function(x):
    Fx = F(x)
    result = np.zeros_like(x)
    for i in range(13, 26):
        comb_val = math.comb(25, i)
        result += comb_val * np.exp(-x) * (i * F(x) ** (i - 1) * (1 - F(x)) ** (n - i) - (F(x) ** i * (n - i) * (1 - F(x)) ** (n - i - 1)))
    
    return result

n = 25
x = np.random.exponential(1, n) # Генерируем выборку

# a) --------------------

# Ищем все моды:
values, counts = np.unique(x, return_counts=True)
max_count = np.max(counts)
moda = values[counts == max_count]

# Выводим результат:
moda_ans = f'\na)\tМода: '
if (moda.size == counts.size): moda_ans += 'Все элементы являются модой'
else: 
    for m in moda: moda_ans += f'{m} '
    
print(moda_ans)

print(f'\tМедиана: {np.median(x)}')
print(f'\tРазмах: {np.max(x) - np.min(x)}')
print(f'\tКоэффициент ассиметрии: {asym_coeff(x)}\n')

# b) --------------------

# Строим ЭФР(ECDF)
x_ecdf = np.sort(x)
y_ecdf = np.arange(1, len(x_ecdf) + 1) / len(x_ecdf)

# Отображаем с помощью matplotlib
x_plot = np.concatenate([[x_ecdf[0] - 1], x_ecdf, [x_ecdf[-1] + 1]])
y_plot = np.concatenate([[0], y_ecdf, [1]])

plt.figure(figsize=(10, 6))
plt.step(x_plot, y_plot, where='post', label='ЭФР', color='red')
plt.scatter(x_ecdf, y_ecdf, s=20, alpha=0.5, label='Выколотые точки')
plt.xlabel('x')
plt.ylabel('F(x)')
plt.title('Эмпирическая функция распределения')
plt.grid(True, alpha=0.3)
plt.ylim(-0.1, 1.1)
plt.xlim(x_ecdf[0] - 1, x_ecdf[-1] + 1)
plt.legend()
plt.savefig('b) ECDF')

# Строим гистограмму
plt.figure(figsize=(10, 6))
plt.hist(x, bins=10, color='skyblue', edgecolor='black')
plt.xlabel('Значения')
plt.ylabel('Частота')
plt.title('Гистограмма')
plt.grid(True, alpha=0.3)
plt.savefig('b) Hist')

# Строим Boxplot
plt.figure(figsize=(10, 6))
plt.boxplot(x, patch_artist=True, boxprops=dict(facecolor='lightblue'), vert=False)
plt.ylabel('Значения')
plt.title('Boxplot')
plt.grid(True, alpha=0.3, axis='y')
plt.savefig('b) Boxplot')

print(f"b)\tЭФР: 'ECDF.png'\n\tГистограмма: 'Hist.png'\n\tBoxplot: 'Boxplot.png'\n")

# c) --------------------

# По ЦПТ x_ср ---> N(1, 1/n)

plt.figure(figsize=(10, 6))
mean = 1
sigma = np.sqrt(1 / n)  # стандартное отклонение

# Формула плотности нормального распределения
x_clt = np.linspace(-2, 4, 1000)
y_clt = (1 / (sigma * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((x_clt - mean) / sigma)**2)

plt.plot(x_clt, y_clt, color='red', label='ЦПТ')
plt.xlabel('x')
plt.ylabel('Плотность вероятности')
plt.title('ЦПТ и Bootstrap')
plt.xlim(0.25, 1.75)
plt.grid(True, alpha=0.3)

# Bootstrap
boostrap_samples = np.random.choice(x, (1000, n), replace=True)
bootstrap_points = np.mean(boostrap_samples, axis=1)
plt.hist(bootstrap_points, bins=100, color='skyblue', edgecolor='black', density=True, label='Bootstrap')

plt.legend()
plt.savefig("c) CLT and Bootstrap")

print("c)\tГрафики: 'CLT and Bootstrap.png'\n\tВидим несоответствие, т.к. ЦПТ плохо работает при малых n\n")

# d) --------------------

# Bootstrap
boostrap_samples = np.random.choice(x, (1000, n), replace=True)
bootstrap_points = np.apply_along_axis(asym_coeff, axis=1, arr=boostrap_samples)

plt.figure(figsize=(10, 6))
counts, bins, patches = plt.hist(bootstrap_points, bins=100, color='skyblue', edgecolor='black', density=True, label='Bootstrap')

bin_width = bins[1] - bins[0]
idx = np.searchsorted(bins, 1)
counts_left = counts[:idx]
prob = np.sum(counts_left) * bin_width

for i in range(idx):
    plt.bar(bins[i], counts[i], width=bin_width, 
           color='yellow', alpha=0.5, align='edge')

plt.axvline(x=1, color='red', linestyle='--', linewidth=2,
            label=f'x = 1 (Площадь слева = {prob})')

plt.xlabel('x')
plt.ylabel('Плотность вероятности')
plt.legend()
plt.title('Коэффициент ассиметрии')
plt.savefig("d) Asym coeff")

print(f"d)\tГрафик: 'Asym coeff.png'\n\tВероятность, что КА < 1: {prob}\n")

# e) --------------------

plt.figure(figsize=(10, 6))

# Формула плотности нормального распределения
x_theory = np.linspace(-2, 4, 1000)
y_theory = med_function(x_theory)

plt.plot(x_theory, y_theory, color='red', label='13-я статистика')
plt.xlabel('x')
plt.ylabel('Плотность вероятности')
plt.title('Распределение медианы и Bootstrap')
# plt.xlim(0.25, 1.75)
plt.grid(True, alpha=0.3)

# Bootstrap
boostrap_samples = np.random.choice(x, (1000, n), replace=True)
bootstrap_points = np.median(boostrap_samples, axis=1)
plt.hist(bootstrap_points, bins=100, color='skyblue', edgecolor='black', density=True, label='Bootstrap')
plt.xlim(-0.5, 2)
plt.legend()
plt.savefig("e) Median")

print("e)\tГрафик: 'e) Median.png'\n")
