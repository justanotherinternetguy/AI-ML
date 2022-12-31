# ORDINARY LEAST SQUARE - find b0 and b1 weights (coe + bias coe)
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
plt.rcParams['figure.figsize'] = (20.0, 10.0)

data = pd.read_csv('./headbrain.csv')
print(data.shape)
# print(data.head())

# need to find LINEAR relationship between Head Size and Brain Weights
X = data['Head Size(cm^3)'].values
Y = data['Brain Weight(grams)'].values

mean_x = np.mean(X)
mean_y = np.mean(Y)
n = len(X)

num = 0
deno = 0

for i in range(n):
    num += (X[i] - mean_x) * (Y[i] - mean_y)
    deno += (X[i] - mean_x) ** 2

b1 = num / deno
b0 = mean_y - (b1 * mean_x)

# ROOT MEAN SQUARED ERROR
rmse = 0
for i in range(n):
    y_pred = b0 + b1 * X[i]
    rmse += (Y[i] - y_pred) ** 2

rmse = np.sqrt(rmse/n)
print(rmse)

# find R^2

sum_squares = 0
sum_remain = 0

for i in range(n):
    y_pred = b0 + b1 * X[i]
    sum_squares += (Y[i] - mean_y) ** 2
    sum_remain += (Y[i] - y_pred) ** 2

r_squared = 1 - (sum_remain / sum_squares)
print(r_squared)

plt.scatter(X, Y)
plt.show()
