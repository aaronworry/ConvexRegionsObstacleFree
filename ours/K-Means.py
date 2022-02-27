import numpy as np
from getTestData import getData
import matplotlib.pyplot as plt
import cvxpy as cp

### k-means + 最小二乘法：不能保证收敛
### k-means + 总距离最近（使集合的点到直线的距离之和最小）： ？

def min_distance(dim, data):
    A = cp.Variable(dim)
    b = 5 * np.random.randn(1)
    p = np.array(data)

    cost = cp.sum_squares(p @ A - b)
    objective = cp.Minimize(cost)
    constraints = []

    prob = cp.Problem(objective, constraints)
    prob.solve(verbose=True)
    # return C.value, d.value, prob.value
    return A.value, b, prob.value




def getLines(n, dim, data):
    """
    get n lines in dim = 2
    :param n: number of lines
    :param data: obstacle position    [dim, m]
    :return: n lines:  a_n , b_n
    """
    #init
    beta = np.random.randn(n, dim)   # [n, 2]
    print(beta)
    kkk = 0
    while kkk < 20:
        kkk += 1
        # 求每个点到n的直线的距离，点归属于距离最近的
        cluster = get_n_cluster(n, dim, beta, data)
        # 根据重新分类的组合，计算n条直线
        for i in range(n):
            beta[i] = least_squares(dim, np.array(cluster[i]).T)
            """
            if len(cluster[i]):
                A, b, _ = min_distance(dim, cluster[i])
                beta[i][0], beta[i][1] = -1 * b / A[1], -1 * A[0] / A[1]
            """
    return beta, cluster


    # g = computeLine(n, data)

def get_n_cluster(n, dim, beta, data):
    cluster = [[] for _ in range(n)]
    for i in range(len(data[0])):
        index = np.argmin([np.abs(data[:, i][0] * item[1] + item[0] - data[:, i][1]) / np.sqrt(1 + item[1] ** 2) for item in beta])
        cluster[index].append(data[:, i])
    return cluster


def least_squares(dim, data):
    """
    最小二乘拟合直线
    :param dim:
    :param data: [dim, m]
    :return:   beta_0 + beta_1 * x = y
    """
    X, Y = data[0], data[1]
    temp = np.array([1] * len(X))
    X = np.vstack((temp, X))
    beta = np.linalg.inv(X.dot(X.T)).dot(X).dot(Y)
    return beta


# 画图
# beta  [4, 2]   type: np.array
fig = plt.figure()
bx = fig.add_subplot(121)
data = getData()
beta, cluster = getLines(4, 2, data)
x = data[0]
y = data[1]
bx.scatter(x, y, color='b')
ax = fig.add_subplot(122)
X = [-2, 2]
y0 = [beta[0][0] - 2 * beta[0][1], beta[0][0] + 2 * beta[0][1]]
y1 = [beta[1][0] - 2 * beta[1][1], beta[1][0] + 2 * beta[1][1]]
y2 = [beta[2][0] - 2 * beta[2][1], beta[2][0] + 2 * beta[2][1]]
y3 = [beta[3][0] - 2 * beta[3][1], beta[3][0] + 2 * beta[3][1]]

ax.scatter(np.array(cluster[0]).T[0], np.array(cluster[0]).T[1], color='r')
ax.scatter(np.array(cluster[1]).T[0], np.array(cluster[1]).T[1], color='g')
ax.scatter(np.array(cluster[2]).T[0], np.array(cluster[2]).T[1], color='b')
ax.scatter(np.array(cluster[3]).T[0], np.array(cluster[3]).T[1], color='y')

ax.plot(X, y0, color='c')
ax.plot(X, y1, color='c')
ax.plot(X, y2, color='c')
ax.plot(X, y3, color='c')

plt.show()