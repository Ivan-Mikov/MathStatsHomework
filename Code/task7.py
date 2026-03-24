import math
from scipy.stats import chi2


l = 0.61
n = 200
def P(k): 
    p = (l ** k) / (math.factorial(k)) * math.exp(-l)
    print(f"P{k} = {p}")
    return p


p = [0] * 6
P5 = 1
for k in range(5):
    p[k] = P(k)
    P5 -= p[k]
p[5] = P5
print(f"P{5} = {P5}")

p_new = [0] * 4
p_new[0] = p[0]
p_new[1] = p[1]
p_new[2] = p[2]
p_new[3] = p[3] + p[4] + p[5]

m = [109, 65, 22, 4]
delta = 0
for k in range(4):
    delta += ((m[k] - n * p_new[k]) ** 2) / (n * p_new[k])

# print(f"delta = {delta}")

# df = 3

# x = 0.7054

# result = chi2.sf(x, df)

# print(f"P(χ²(3) > {x}) = {result}")