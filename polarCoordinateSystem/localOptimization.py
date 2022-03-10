import numpy as np
import math
from scipy.stats import multivariate_normal


def find_last_hyperplanes(points, new_points, hyperplances):
    """
    :param new_points:  array [n, 2]
    :param hyperplances: list [hp]    m
    :return:
    """
    last_hyperplanes = [-1 * item for item in hyperplances]
    while whetherEnd(last_hyperplanes, hyperplances):
        last_hyperplanes = hyperplances
        weight = cal_weight(new_points, last_hyperplanes)
        hyperplances = update_hyperplanes(points, weight)
    return hyperplances, weight

def whetherEnd(last_hyperplanes, hyperplances):
    n = len(last_hyperplanes)
    maxDistance = -1.
    for i in range(n):
        x1, y1 = last_hyperplanes[i][0] * np.cos(last_hyperplanes[i][1])
        x2, y2 = hyperplances[i][0] * np.cos(hyperplances[i][1])
        distance = np.sqrt((x1-x2)**2 + (y1 - y2)**2)
        if distance >= maxDistance:
            maxDistance = distance
    if maxDistance > 1.:
        return True
    else:
        return False


def cal_weight(new_points, hyperplances):
    n, m = len(new_points), len(hyperplances)
    result = np.zeros((n, m))
    for i in range(n):
        for j in range(m):
            distance = abs(hyperplances[j][0] - new_points[i][0] * np.cos(hyperplances[j][1] - new_points[i][1]))
            result[i][j] = np.exp(-1 * distance)
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




