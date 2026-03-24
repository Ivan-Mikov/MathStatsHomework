import numpy as np

theta = 10
n = 100
xn = theta + theta * np.random.random(n)


def phi(t): return n * (t - 1) ** (n - 1)


# Точный метод
x_max = np.max(xn)
t1 = 1 + (0.025) ** (1 / n)
t2 = 1 + (0.975) ** (1 / n)
exact_left = x_max / t2
exact_right = x_max / t1
print(f'Точный метод: ({(exact_left):.3f}, {(exact_right):.3f})')

# Ассимптотический (ОММ)
alpha_1 = np.mean(xn)
alpha_2 = np.mean(xn ** 2)
theta_1 = 2 / 3 * alpha_1
t1 = -1.96
t2 = 1.96
assym_left = theta_1 - 4 / 9 * t2 * np.sqrt((alpha_2 - alpha_1 ** 2) / n)
assym_right = theta_1 - 4 / 9 * t1 * np.sqrt((alpha_2 - alpha_1 ** 2) / n)
print(f'Ассимптотический метод (ОММ): ({(assym_left):.3f}, {(assym_right):.3f})')

# Непараметрический бутстрап
boostrap_samples = np.random.choice(xn, (1000, n), replace=True)
bootstrap_points = np.sort(
    np.apply_along_axis(
        lambda x: 2 / 3 * np.mean(x) - theta_1,
        axis=1,
        arr=boostrap_samples
    )
)
k1 = 24
k2 = 974
bootstrap_left = theta_1 - bootstrap_points[k2]
bootstrap_right = theta_1 - bootstrap_points[k1]
print(f'Непараметрический Bootstrap: ({(bootstrap_left):.3f}, {(bootstrap_right):.3f})')

print(f'''
Длины интервалов:
Точный метод: {exact_right - exact_left},
Ассимптотический метод (ОММ): {assym_right - assym_left},
Bootstrap: {bootstrap_right - bootstrap_left}
''')
