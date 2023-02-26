import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import seaborn as sns; sns.set()
import numpy as np

def gen_data(m, b, seed, error, length):
    """Returns two arrays of randomly generated and fitted numbers for linear regression"""
    rng = np.random.RandomState(seed)
    X = 10 * rng.rand(length)
    Y = m * X - b + (error*rng.randn(length))
    return X, Y

# pred: y = b0 + b1 * x


# RMSE
from math import sqrt
def rmse(valid, pred):
    """
    valid -- actual data
    pred -- predicted data
    """
    sum_err = 0.0
    for i in range(len(valid)):
        pred_err = pred[i] - valid[i]
        sum_err += (pred_err ** 2)
        mean_err = sum_err / float(len(valid))
    return sqrt(mean_err)

def cov(x, mean_x, y, mean_y):
    covar = 0.0
    for i in range(len(x)):
        covar += (x[i] - mean_x) * (y[i] - mean_y)
    return covar

def var(vals, mean):
    return sum([(x-mean) ** 2 for x in vals])

print(rmse([3], [4]))

X, Y = gen_data(2, -5, 1, 0.5, 50)
plt.scatter(X, Y)
plt.show()
