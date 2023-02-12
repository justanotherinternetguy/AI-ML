import numpy as np 
import matplotlib.pyplot as plt 

start_point = -1
end_point = 6
num_points = 200

X = np.linspace(start_point,end_point,num_points)

errs = 6*np.random.randn(num_points)

data = 5 + 3.5*X + 2*X*X + errs 

plt.figure() 
plt.plot(data,'X')
plt.show() 

