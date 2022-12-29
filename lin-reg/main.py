import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from scipy import stats


X = [0.4, 1, 3, 7.2, 10, 12, 26.6, 26.8, 28, 28.4, 38, 42.4, 43.4, 47.9, 55, 56.2, 65]
Y = [0, 175.44, 315.79, 654.39, 857.90, 1066.67, 1500, 1850.88, 2010.53, 2140.36, 2756.15, 2887.738, 2945.63, 3107.03, 3571.94, 3966.68, 4352.64]


# PLOT #
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

# END OF PLOT #

# ROOT MEAN SQUARED ERROR #
rmse = 0
y_pred = b0 + b1 * X[i]

for i in range(n):
    rmse += (Y[i] - y_pred) ** 2

rmse = np.sqrt(rmse/n)
print("RMSE: ", rmse)

# R^2 score, 0-1
sum_squares = 0
sum_res = 0

for i in range(n):
    sum_squares += (Y[i] - y_mean) ** 2
    sum_res += (Y[i] - y_pred) ** 2

acc = 1 - (sum_squares / sum_res)
print("%: ", acc)
