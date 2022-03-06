"""
使用极坐标，对每一个theta下的直线的距离求一个高斯分布，根据方差和包含的点数排序。
排序的第一个就是第一个超平面的theta，然后去掉这个theta包含的点（排序的列表每一项都去掉这些点），再找下一个。
依次类推


找到一个超平面拟合结果作为初始值

然后使用GMM和谱聚类的思想在初值附近找一个局部最优解  ： 已有的 compatible cluster merging
"""





"""
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
"""
