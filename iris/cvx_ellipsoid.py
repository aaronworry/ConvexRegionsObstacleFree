import numpy as np
import cvxpy as cp

def cvx_ellipsoid(A, b):
    dim = len(A[0])
    m = len(b)
    C = cp.Variable((dim, dim), symmetric=True)
    d = cp.Variable(dim)
    objective = cp.Maximize(cp.atoms.log_det(C))
    constraints = [C >> 0.]
    constraints += [cp.atoms.norm(C @ A[i].T, 2) + A[i] @ d <= b[i] for i in range(m)]

    prob = cp.Problem(objective, constraints)
    prob.solve()
    return C.value, d.value, prob.value


if __name__ == '__main__':
    #Vertices of a pentagon in 2D
    p = [[0., 0.], [1., 3.], [5.5, 4.5], [7., 4.], [7., 1.], [3., -2.]]
    nVerts = len(p)

    #The hyperplane representation of the same polytope
    A = [[-p[i][1] + p[i - 1][1], p[i][0] - p[i - 1][0]]
         for i in range(len(p))]
    b = [A[i][0] * p[i][0] + A[i][1] * p[i][1] for i in range(len(p))]
    A = np.array(A)
    b = np.array(b)

    # Po, co = lownerjohn_outer(p)
    Ci, di, volume = cvx_ellipsoid(A, b)
    print(Ci, di, volume)

    #Visualization
    try:
        import numpy as np
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        import matplotlib.patches as patches

        #Polygon
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.add_patch(patches.Polygon(p, fill=False, color="red"))
        #The inner ellipse
        theta = np.linspace(0, 2 * np.pi, 100)
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
    except:
        print("Inner:")
        print("  C = ", Ci)
        print("  d = ", di)
        # print("Outer:")
        # print("  P = ", Po)
        # print("  c = ", co)