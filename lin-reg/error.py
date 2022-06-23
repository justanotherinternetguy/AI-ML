import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

x = [5, 7, 8, 7, 2, 17, 2, 9, 4, 11, 12, 9, 6]
y = [99, 86, 87, 88, 111, 86, 103, 87, 94, 78, 77, 85, 86]



xpoints = np.arange(start=1, stop=50)
ypoints = np.arange(start=1, stop=50)

# points
plt.scatter(x, y)
# line
plt.plot(xpoints, ypoints)
plt.show()
