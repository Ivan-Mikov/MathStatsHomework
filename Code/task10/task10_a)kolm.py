import numpy as np


def F_left(x, xn): return np.sum(xn < x) / len(xn)
def F_right(x, xn): return np.sum(xn <= x) / len(xn)


def sup(xn, F):
    sup = 0
    for x in xn:
        first = abs(F_left(x, xn) - F(x))
        second = abs(F_right(x, xn) - F(x))
        sup = max(sup, first, second)
    return sup


n = 100
vals = np.arange(10)
m = np.array([5, 8, 6, 12, 14, 18, 11, 6, 13, 7])
xn = np.repeat(vals, m)

alpha1 = np.mean(xn)
alpha2 = np.mean(xn ** 2)

theta2 = alpha1 + np.sqrt(4 * alpha1 ** 2 - alpha2)
theta1 = alpha1 - np.sqrt(4 * alpha1 ** 2 - alpha2)

delta_ = np.sqrt(n) * sup(xn, lambda x: (x - theta1) / (theta2 - theta1))

N = 50000
delta = [0] * N
for i in range(N):
    xn_star = np.random.uniform(theta1, theta2, n)

    alpha1_star = np.mean(xn_star)
    alpha2_star = np.mean(xn_star ** 2)

    theta2_star = alpha1_star + np.sqrt(4 * alpha1_star ** 2 - alpha2_star)
    theta1_star = alpha1_star - np.sqrt(4 * alpha1_star ** 2 - alpha2_star)

    delta[i] = np.sqrt(n) * sup(
        xn,
        lambda x: (x - theta1_star) / (theta2_star - theta1_star)
    )

delta_sort = np.sort(delta)
count = np.sum(delta_sort >= delta_)
print(count / N)
