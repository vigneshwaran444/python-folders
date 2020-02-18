import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
x = np.arange(0,100)
y = np.arange(0,100)
print (x)
print (y)
lr = LinearRegression()

#lr.fit(x,y)
# you should see an error...

x.ndim
y.ndim

x.shape
y.shape

x = x.reshape(-1,1)
x.shape

x.ndim

lr.fit(x,y)
# now it works!

plt.scatter(x, y, color = 'red')

plt.plot(x, lr.predict(x), color = 'blue')
plt.title('Linear Regression Demo')
plt.xlabel('X')
plt.ylabel('Y')
plt.show()
