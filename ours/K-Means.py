import numpy as np
from getTestData import getData

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
    while kkk < 15:
        kkk += 1
        # 求每个点到n的直线的距离，点归属于距离最近的
        cluster = get_n_cluster(n, dim, beta, data)
        # 根据重新分类的组合，计算n条直线
        for i in range(n):
            beta[i] = computeLine(dim, np.array(cluster[i]).T)
        print(beta)
    return beta


    # g = computeLine(n, data)

def get_n_cluster(n, dim, beta, data):
    cluster = [[] for _ in range(n)]
    for i in range(len(data[0])):
        index = np.argmin([np.abs(data[:, i][0] * item[1] + item[0] - data[:, i][1]) / np.sqrt(1 + item[1] ** 2) for item in beta])
        cluster[index].append(data[:, i])
    return cluster


def computeLine(dim, data):
    """

    :param dim:
    :param data: [dim, m]
    :return:
    """
    X, Y = data[0], data[1]
    temp = np.array([1] * len(X))
    X = np.vstack((temp, X))
    beta = np.linalg.inv(X.dot(X.T)).dot(X).dot(Y)
    return beta

data = getData()
beta = getLines(4, 2, data)
print(beta, np.shape(beta))
# 画图
# beta  [4, 2]   type: np.array