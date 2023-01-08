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

X = np.array([math, read]).T
Y = np.array(write)

reg = LinearRegression()
reg = reg.fit(X, Y)
Y_pred = reg.predict(X)

rmse = np.sqrt(mean_squared_error(Y, Y_pred))
r2 = reg.score(X, Y)
print(rmse)
print(r2)
