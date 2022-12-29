import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from scipy import stats

X = np.random.randint(100, size=100)
Y = np.random.randint(100, size=100)


x_mean = np.mean(X)
y_mean = np.mean(Y);
n = len(X)

numerator = 0
denominator = 0
for i in range(n):
    numerator += (X[i] - x_mean) * (Y[i] - y_mean)
    denominator += (X[i] - x_mean) ** 2


b1 = numerator / denominator
b0 = y_mean - (b1 * x_mean)

x_max = np.max(X) + 100
x_min = np.min(X) - 100

x = np.linspace(x_min, x_max, 1000)
y = b0 + b1 * x

plt.plot(x, y, color='#00ff00')
plt.scatter(X, Y)
plt.show()
