# 类似混合聚类
"""
初始化 n 条直线
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
