import numpy as np
import scipy.stats as stats


def F_left(x, xn): return np.sum(xn < x) / len(xn)
def F_right(x, xn): return np.sum(xn <= x) / len(xn)


# def sup(xn, F):
#     sup = 0
#     for x in xn:
#         first = abs(F_left(x, xn) - F(x))
#         second = abs(F_right(x, xn) - F(x))
#         sup = max(sup, first, second)
#     return sup

def sup(xn, F):

    unique_x = np.unique(xn)
    n = len(xn)

    left_counts = np.array([np.sum(xn < x) for x in unique_x])
    right_counts = np.array([np.sum(xn <= x) for x in unique_x])

    left_ecdf = left_counts / n
    right_ecdf = right_counts / n

    F_vals = F(unique_x)

    left_diff = np.abs(left_ecdf - F_vals)
    right_diff = np.abs(right_ecdf - F_vals)

    return np.max(np.maximum(left_diff, right_diff))


n = 100
vals = np.arange(10)
m = np.array([5, 8, 6, 12, 14, 18, 11, 6, 13, 7])
xn = np.repeat(vals, m)

alpha1 = np.mean(xn)
alpha2 = np.mean(xn ** 2)

theta1 = alpha1
theta2 = alpha2 - alpha1 ** 2

delta_ = np.sqrt(n) * sup(
    xn,
    lambda x: stats.norm.cdf(x, loc=theta1, scale=np.sqrt(theta2))
)

N = 50000
delta = [0] * N
for i in range(N):
    xn_star = np.random.normal(theta1, np.sqrt(theta2), n)

    alpha1_star = np.mean(xn_star)
    alpha2_star = np.mean(xn_star ** 2)

    theta1_star = alpha1_star
    theta2_star = alpha2_star - alpha1_star ** 2

    delta[i] = np.sqrt(n) * sup(
        xn_star,
        lambda x: stats.norm.cdf(x, loc=theta1_star, scale=np.sqrt(theta2_star))
    )


count = np.sum(delta >= delta_)
print(count / N)
