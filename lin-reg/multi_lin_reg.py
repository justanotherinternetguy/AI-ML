# batch grad desc
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

plt.rcParams['figure.figsize'] = (20.0, 10.0)
from mpl_toolkits.mplot3d import Axes3D

data = pd.read_csv("student.csv")

math = data['Math'].values
read = data['Reading'].values
write = data['Writing'].values

fig = plt.figure()
ax = Axes3D(fig)
ax.scatter(math, read, write, color="#FF0000")
plt.show()

m = len(math)
x0 = np.ones(m)
X = np.array([x0, math, read]).T

# init coe
B = np.array([0, 0, 0])
Y = np.array(write)
alpha = 0.0001

def cost_function(X, Y, B):
    m = len(Y)
    J = np.sum((X.dot(B) - Y) ** 2) / (2 * m)
    return J

inital_cost = cost_function(X, Y, B)
print(inital_cost) # initial cost is huge, we need to reduce with GD

def grad_desc(X, Y, B, alpha, iters):
    cost_history = [0] * iters
    m = len(Y)

    for iteration in range(iters):
        h = X.dot(B)
        loss = h - Y
        gradient = X.T.dot(loss) / m
        B = B - alpha * gradient

        cost = cost_function(X, Y, B)
        cost_history[iteration] = cost

    return B, cost_history

# after 100,000 iters
newB, cost_history = grad_desc(X, Y, B, alpha, 100000)
print(newB)
print(cost_history[-1])


# RMSE eval
def rmse(Y, Y_pred):
    rmse = np.sqrt(sum((Y - Y_pred) ** 2) / len(Y))
    return rmse

# R2 eval
def r2_score(Y, Y_pred):
    mean_y = np.mean(Y)
    ss_tot = sum((Y - mean_y) ** 2)
    ss_res = sum((Y - Y_pred) ** 2)
    r2 = 1 - (ss_res / ss_tot)
    return r2

Y_pred = X.dot(newB)
print(rmse(Y, Y_pred))
print(r2_score(Y, Y_pred))
