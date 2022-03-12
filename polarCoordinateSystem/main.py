"""
定义部分：
超平面的表征：先找到平面上距离原点O最近的一点A，向量OA为超平面的表征。
距离rho: norm2(OA)
方向:  2D:   theta1 = arcsin( OA[1] / norm2(OA) )，  即投影到x轴
      3D:   theta2 = arcsin( OA[2] / norm2(OA) ), theta1 = arcsin( OA[1] / norm2(OA去除最后一位) )， 即先投影到xOy平面，再投影到x轴

2D，每个点可能的直线是一个周期为2pi的cos函数，theta=theta1时，函数取最大值rho。每个周期内：函数的定义域为[theta1 - 90°， theta1 + 90°]，值域为[0, rho]


算法部分：
假设：直线的噪声不超过某个值sigma，直线的点数最少为某个数num，最多有直线数m

找到一个超平面拟合结果作为初始值：
    使用极坐标，对每一个theta下的直线的距离：
    1.求均值和方差，用宽度为w的两个平面去夹这些点，如果特别不合理，则在原点的同一边存在平行线。将这个情况的theta放入一个列表.
    2.如果不含平行线，根据方差和点数，构建另一个theta列表.
    3.对于1中的theta列表,根据每个theta可能包含的点的数目对theta排列（大到小）,对于排序后的列表的每个theta
        3.1使用多峰分布（2,3,4，。。。）拟合，找到最好的拟合结果   -> 多个聚类
        3.2根据拟合的结果，将这些结果放到result中，根据这个超平面的宽度标记出属于这些平面的点
        3.3步骤3和2中的theta列表都移除3.2中的这些点，对步骤3的theta列表重新排序，跳转到3.1
            3.4如何跳出步骤3？
    4.对于2中的theta:
        4.1排序的第一个就是第一个超平面，将其放入result中，根据这个超平面的宽度标记出属于这个平面的点。
        4.2列表中剩下的theta中移除2.1中被标记的点，调到2.1找下一个平面。
            4.3如果找的平面的特征，和上一个找到的平面差距特别大，即不能认为现在的这个平面是一个合理的。

根据特征，选择是否合并某些直线?

然后使用GMM和谱聚类的思想在初值附近找一个局部最优解  ： 已有的 compatible cluster merging
    谱聚类 + GMM的思想

    极坐标下找到的n个直线，作为初始解
    求m个点与直线的距离  [m, n]
    设点到直线的距离d ~ N(0, sigma**2)
    直线参数 [beta0, beta1]
    Sigma [1, n]
    点属于直线的概率  Weight     [m, n]
    n条直线的点的数目占比   Pi   [1, n]

    E-step: 求Weight
    M-step：   求 Pi, 直线参数， Sigma

    收敛条件： Sigma越小越好

根据特征，选择是否合并某些直线?
"""
import numpy as np
from findInitialHyperplanes import get_initial_hyperplanes, Point2D
from localOptimization import find_last_hyperplanes
from getTestData import getData, getData2, getData3
from Kmeans import getLines
import matplotlib.pyplot as plt

def conc(hyperplanes):
    pass


def get_n_cluster(Weight, data):
    cluster = [[] for _ in range(len(Weight[0]))]
    for i in range(len(data[0])):
        index = np.argmax(Weight[i])
        cluster[index].append(data[:, i])
    return cluster

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
    delta = np.sqrt(B * B - 4 * A * C)
    k1, k2 = (delta - B) / (2 * A), (-1 * delta - B) / (2 * A)
    beta = np.array([y_avr - k1 * x_avr, k1])
    return beta

def draw(data, pre_hyperplances, cluster):
    # 3 个图片
    # 1:所有点， 2：所有点以及pre_hyperplances对应的点（mu.cos\theta, mu.sin\theta）， 3：分类好的点和线
    fig = plt.figure()
    bx = fig.add_subplot(121)
    x = data[0]
    y = data[1]
    bx.scatter(x, y, color='b')
    bx.scatter(0, 0, color='r')
    for item in pre_hyperplances:
        x = item[0] * np.cos(item[1])
        y = item[0] * np.sin(item[1])
        bx.scatter(x, y, color='r')
    ax = fig.add_subplot(122)
    colors = ['r', 'g', 'b', 'y', 'c', 'm', 'k']
    for i in range(len(cluster)):
        if len(cluster[i]) > 0:
            ax.scatter(np.array(cluster[i]).T[0], np.array(cluster[i]).T[1], color=colors[i%len(colors)])
            # beta = min_distance(np.array(cluster[i]).T)
            # X = [-2, 2]
            # y = [beta[0] - 2 * beta[1], beta[0] + 2 * beta[1]]
            # ax.plot(X, y, color=colors[i])

    plt.show()

if __name__ == "__main__":
    import math
    data = getData3(0.1)
    points = data.T
    new_points = [Point2D(item) for item in points]
    pre_hyperplances = get_initial_hyperplanes(new_points, resolution = 5, maxSigma=0.2, minNumber=30, maxHyperplanes=4)
    # pre_hyperplances = np.array([[10/np.sqrt(37), math.atan(6) + np.pi/2],[9/np.sqrt(5), math.atan(2)-np.pi/2],[8/np.sqrt(10), math.atan(-3)+np.pi/2],[7/np.sqrt(2), math.atan(-1)-np.pi/2]])
    points_data = np.array([item.date_in_polar for item in new_points])
    # hyperplanes, Weight = find_last_hyperplanes(points, points_data, pre_hyperplances, maxSigma=0.15)
    # cluster = get_n_cluster(Weight, data)
    # 合并 ？
    beta, cluster = getLines(pre_hyperplances, data)
    draw(data, pre_hyperplances, cluster)


