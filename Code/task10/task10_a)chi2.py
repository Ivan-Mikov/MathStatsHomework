import numpy as np
import scipy.stats as stats


n = 100
vals = np.arange(10)
m = np.array([5, 8, 6, 12, 14, 18, 11, 6, 13, 7])

theta1 = 0
theta2 = float(850 / 93)

p = [0] * 10
p[0] = 0.5 / theta2
p[9] = (theta2 - 8.5) / theta2
for i in range(1, 9):
    p[i] = 1 / theta2

p = np.array(p)

delta = np.sum((m - n * p) ** 2 / (n * p))
print(delta)

result = stats.chi2.sf(delta, 7)

print(f"{result:.10f}")
