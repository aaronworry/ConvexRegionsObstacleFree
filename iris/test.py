import numpy as np
from iris import IRISProblem, IRISOptions, IRISDebugData, inflate_region
import matplotlib.pyplot as plt
import matplotlib.patches as patches

problem = IRISProblem(2)
problem.setSeedPoint(np.array([0.1, 0.1]))
problem.addObstacle(np.array([[0., 1., 5.5, 7., 7., 3.], [0., 3., 4.5, 4., 1., -2]]))

options = IRISOptions()

debug = IRISDebugData()

polyhedron, ellipsoid = inflate_region(problem, options, debug)


fig = plt.figure()
ax = fig.add_subplot(111)
p = [[0., 0.], [1., 3.], [5.5, 4.5], [7., 4.], [7., 1.], [3., -2.]]
ax.add_patch(patches.Polygon(p, fill=False, color="red"))
#The inner ellipse
theta = np.linspace(0, 2 * np.pi, 100)
Ci, di = ellipsoid.C_, ellipsoid.d_
x = Ci[0][0] * np.cos(theta) + Ci[0][1] * np.sin(theta) + di[0]
y = Ci[1][0] * np.cos(theta) + Ci[1][1] * np.sin(theta) + di[1]
ax.plot(x, y)
#The outer ellipse
# x, y = np.meshgrid(np.arange(-1.0, 8.0, 0.025), np.arange(-3.0, 6.5, 0.025))
# ax.contour(x, y, (Po[0][0] * x + Po[0][1] * y - co[0])**2 + (Po[1][0] * x + Po[1][1] * y - co[1])**2, [1])
# ax.autoscale_view()
ax.xaxis.set_visible(False)
ax.yaxis.set_visible(False)
fig.savefig('ellipsoid.png')

