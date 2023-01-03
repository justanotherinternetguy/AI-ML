# ORDINARY LEAST SQUARE - find b0 and b1 weights (coe + bias coe)
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

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

max_x = np.max(X) + 100
min_x = np.min(X) - 100

x = np.linspace(np.min(X) - 100, np.max(X) + 100, 1000)
y = b0 + b1 * x

plt.plot(x, y, color="#ff0000")
plt.scatter(X, Y)
plt.show()

X = x.reshape((n, 1))
reg = LinearRegression()
reg = reg.fit(X, Y)
Y_pred = reg.predict(X)

mse = mean_squared_error(Y, Y_pred)
rmse = np.sqrt(mse)
r2_score = reg.score(X, Y)
print(rmse)
print(r2_score)
