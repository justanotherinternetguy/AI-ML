import matplotlib.pyplot as plt
from scipy import stats

x = [0.4, 1, 3, 7.2, 10, 12, 26.6, 26.8, 28, 28.4, 38, 42.4, 43.4, 47.9, 55, 56.2, 65]
y = [0, 175.44, 315.79, 654.39, 857.90, 1066.67, 1500, 1850.88, 2010.53, 2140.36, 2756.15, 2887.738, 2945.63, 3107.03, 3571.94, 3966.68, 4352.64]

slope, intercept, relation, p, std_err = stats.linregress(x, y)


def graph(x):
  return slope * x + intercept

model = list(map(graph, x))

print("SLOPE: {0}".format(slope))

plt.scatter(x, y)
plt.plot(x, model)
plt.show()

