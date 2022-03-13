import numpy as np
from getTestData import getData, getData2, getData3
import matplotlib.pyplot as plt

def cal_from_hyperplane(hyperplanes):
    beta = hyperplanes.copy()
    for i in range(len(hyperplanes)):
        beta[i][0], beta[i][1] = hyperplanes[i][0] / np.sin(hyperplanes[i][1]), np.tan(hyperplanes[i][1] + np.pi / 2)
    return beta

def getLines(hyperplane, data):
    """
    get n lines in dim = 2
    :param n: number of lines
    :param data: obstacle position    [dim, m]
    :return: n lines:  a_n , b_n

    """
    #init
    beta = cal_from_hyperplane(hyperplane)
    kkk = 0
    while kkk < 15:
        kkk += 1
        # 求每个点到n的直线的距离，点归属于距离最近的
        cluster = get_n_cluster(beta, data)
        # 根据重新分类的组合，计算n条直线
        for i in range(len(cluster)):
            # beta[i] = least_squares(np.array(cluster[i]).T)
            beta[i] = min_distance(np.array(cluster[i]).T)
    return beta, cluster


    # g = computeLine(n, data)

def get_n_cluster(beta, data):
    cluster = [[] for _ in range(len(beta))]
    for i in range(len(data[0])):
        index = np.argmin([np.abs(data[:, i][0] * item[1] + item[0] - data[:, i][1]) / np.sqrt(1 + item[1] ** 2) for item in beta])
        cluster[index].append(data[:, i])
    result = []
    for item in cluster:
        if len(item) > 0:
            result.append(item)
    return result


def least_squares(data):
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

def min_distance(data):
    """
    拟合直线，与直线距离最小
    """
    x_avr, y_avr = np.mean(data[0]), np.mean(data[1])
    A = 0
    B = 0
    C = 0
    for i in range(len(data[0])):
        x = data[0][i] - x_avr
        y = data[1][i] - y_avr
        A += x * y
        B += x * x - y * y
        C += -1 * x * y
    delta = np.sqrt(B*B - 4 * A * C)
    k1, k2 = (delta - B) / (2 * A), (-1 * delta - B) / (2 * A)
    beta = np.array([y_avr - k1 * x_avr, k1])
    return beta