import numpy as np
import matplotlib.pyplot as plt

np.random.randn()

k, b = [6, 2, -3, -0.8], [10, -9, 8, -7]
x = np.array([-3 + 0.1*i for i in range(61)])
result = []
for i in range(4):
    y = k[i] * x + b[i] + 4. * (np.random.rand(len(x)) - 0.5)
    result.append(np.vstack((x, y)))
data = np.hstack((result[0], result[1], result[2], result[3]))
# testData = data.T

fig = plt.figure()
ax = fig.add_subplot(111)
#The inner ellipse
x = data[0]
y = data[1]
ax.scatter(x, y)
#The outer ellipse
# x, y = np.meshgrid(np.arange(-1.0, 8.0, 0.025), np.arange(-3.0, 6.5, 0.025))
# ax.contour(x, y, (Po[0][0] * x + Po[0][1] * y - co[0])**2 + (Po[1][0] * x + Po[1][1] * y - co[1])**2, [1])
# ax.autoscale_view()
ax.xaxis.set_visible(False)
ax.yaxis.set_visible(False)
fig.savefig('ellipsoid.png')