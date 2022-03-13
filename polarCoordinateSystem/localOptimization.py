import numpy as np
import math
from scipy.stats import multivariate_normal

###

def find_last_hyperplanes(points, new_points, hyperplances, maxSigma):
    """
    :param new_points:  array [n, 2]
    :param hyperplances: list [hp]    m
    :return:
    """
    n, m = len(new_points), len(hyperplances)
    last_weight = np.zeros((n, m))
    while True:

        last_hyperplanes = hyperplances
        weight = cal_weight(new_points, last_hyperplanes, maxSigma)
        hyperplances = update_hyperplanes(points, weight)
        d_weight = weight - last_weight
        print(np.max(d_weight))
        if np.max(d_weight) <= 2.:
            break
    # print(weight, hyperplances)
    return hyperplances, weight




def cal_weight(new_points, hyperplances, maxSigma):
    n, m = len(new_points), len(hyperplances)
    result = np.zeros((n, m))
    for i in range(n):
        for j in range(m):
            distance = np.sqrt((hyperplances[j][0] - new_points[i][0] * np.cos(hyperplances[j][1] - new_points[i][1])) ** 2)
            # 使用某种分布，估计weight, lmm 拉普拉斯分布？
            result[i][j] = 1 / distance
    weight = result / result.sum(axis = 1).reshape(-1, 1)
    return weight


def update_hyperplanes(points, weight):
    result = []
    for i in range(len(weight[0])):
        hp = min_distance(weight[:, i], points)
        result.append(hp)
    return result


def min_distance(weight_cols, points):
    """
    拟合直线，与直线距离最小
    """
    x_avr, y_avr = np.average(points, axis=0, weights=weight_cols)
    total = np.sum(weight_cols)
    A = 0
    B = 0
    C = 0
    for i in range(len(points)):
        x = (points[i][0] - x_avr) * weight_cols[i] / total
        y = (points[i][1] - y_avr) * weight_cols[i] / total
        A += x * y
        B += x * x - y * y
        C += -1 * x * y
    delta = np.sqrt(B * B - 4 * A * C)
    k1, k2 = (delta - B) / (2 * A), (-1 * delta - B) / (2 * A)
    # y_avr = k1 * x_avr + b
    bias = y_avr - k1 * x_avr
    mu = abs(bias) / np.sqrt(k1 ** 2 + 1)
    theta = math.atan(k1)
    if mu > 0:
        theta += np.pi / 2
    elif mu < 0:
        theta -= np.pi / 2
    return np.array([mu, theta])




