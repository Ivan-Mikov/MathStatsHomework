import numpy as np
import scipy.stats as stats
import scipy.optimize as optimize


n = 100
vals = np.arange(10)
m = np.array([5, 8, 6, 12, 14, 18, 11, 6, 13, 7])


def f(params):
    a, sigma2 = params
    p = [0] * 10
    borders = np.arange(9) + 0.5

    p[0] = stats.norm.cdf(0.5, loc=a, scale=np.sqrt(sigma2))
    p[9] = 1 - stats.norm.cdf(8.5, loc=a, scale=np.sqrt(sigma2))
    for i in range(1, 9):
        p[i] = stats.norm.cdf(
            borders[i], loc=a, scale=np.sqrt(sigma2)
        ) - stats.norm.cdf(
            borders[i - 1], loc=a, scale=np.sqrt(sigma2)
        )

    p = np.array(p)

    return -np.sum(m * np.log(p))


xn = np.repeat(vals, m)
bounds = [(-np.inf, np.inf), (0, np.inf)]

result = optimize.minimize(
    f,
    [np.mean(xn), np.mean(xn ** 2) - np.mean(xn) ** 2],
    method='L-BFGS-B',
    bounds=bounds
)

# print(result)

# After searching:
a = 4.79
sigma2 = 7.18

p = [0] * 10
borders = np.arange(9) + 0.5

p[0] = stats.norm.cdf(0.5, loc=a, scale=np.sqrt(sigma2))
p[9] = 1 - stats.norm.cdf(8.5, loc=a, scale=np.sqrt(sigma2))
for i in range(1, 9):
    p[i] = stats.norm.cdf(
        borders[i], loc=a, scale=np.sqrt(sigma2)
    ) - stats.norm.cdf(
        borders[i - 1], loc=a, scale=np.sqrt(sigma2)
    )

p = np.array(p)
delta = np.sum((m - n * p) ** 2 / (n * p))

print(delta)

result = stats.chi2.sf(delta, 7)

print(f"{result:.10f}")
