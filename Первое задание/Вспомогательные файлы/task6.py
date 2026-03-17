import numpy as np

theta = 2
n = 100
xn = np.random.pareto(a=theta-1, size=n) + 1


def m(t): return 2 ** (1 / (t - 1))
def sigma(t): return np.log(2) * m(t) / (t - 1)
def I(t): return 1 / (t - 1) ** 2


# Ассимптотический для медианы
theta_wave = 1 + n / (np.sum(np.log(xn)))
t1 = -1.96
t2 = 1.96
med_left = m(theta_wave) - sigma(theta_wave) / np.sqrt(n) * t2
med_right = m(theta_wave) - sigma(theta_wave) / np.sqrt(n) * t1
print(f'Ассимптотический для медианы: ({(med_left):.4f}, {(med_right):.4f})'э
      )

# Ассимптотический для тета
theta_left = theta_wave - t2 / np.sqrt(n * I(theta_wave))
theta_right = theta_wave - t1 / np.sqrt(n * I(theta_wave))
print(f'Ассимптотический для тета: ({(theta_left):.3f}, {(theta_right):.3f})')

# Непараметрический бутстрап
boostrap_samples = np.random.choice(xn, (1000, n), replace=True)
bootstrap_points = np.sort(
    np.apply_along_axis(
        lambda x: 1 + n / (np.sum(np.log(x))) - theta_wave,
        axis=1,
        arr=boostrap_samples
    )
)
k1 = 24
k2 = 974
bootstrap_left1 = theta_wave - bootstrap_points[k2]
bootstrap_right1 = theta_wave - bootstrap_points[k1]
print(f'Непараметрический Bootstrap: ({(bootstrap_left1):.3f}, {(bootstrap_right1):.3f})')

# Параметрический бутстрап
boostrap_samples = np.random.pareto(a=theta_wave-1, size=(50000, n)) + 1
bootstrap_points = np.sort(
    np.apply_along_axis(
        lambda x: 1 + n / (np.sum(np.log(x))) - theta_wave,
        axis=1,
        arr=boostrap_samples
    )
)
k1 = 1250
k2 = 48750
bootstrap_left2 = theta_wave - bootstrap_points[k2]
bootstrap_right2 = theta_wave - bootstrap_points[k1]
print(f'Параметрический Bootstrap: ({(bootstrap_left2):.3f}, {(bootstrap_right2):.3f})')


print(f'''
Длины интервалов для θ:
Ассимптотический метод: {theta_right - theta_left},
Непараметрический бутстрап: {bootstrap_right1 - bootstrap_left1},
Параметрический бутстрап: {bootstrap_right2 - bootstrap_left2}
''')
