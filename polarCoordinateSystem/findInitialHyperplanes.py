import numpy as np



class Point2D():
    def __init__(self, point):
        """
        :param point: [x, y]
        """
        self.theta = np.atan2(point[1], point[0])
        self.dis = np.linalg.norm(point)
        self.define_domain = [self.theta - np.pi/2, self.theta + np.pi/2]

        # 以下的3个可能没用
        self.domain_num = 0
        self.domain = []
        self.process_domain()

    def get_distance(self, theta):
        distance = np.cos(theta - self.theta)
        if distance >= 0:
            return distance
        else:
            return -1

    def process_domain(self):
        # 可能没用
        if self.theta < -np.pi / 2:
            self.domain_num = 2
            self.domain = [[-np.pi, self.theta + np.pi/2], [np.pi - (-np.pi/2 - self.theta), np.pi]]
        elif self.theta > np.pi / 2:
            self.domain_num = 2
            self.domain = [[-np.pi, -np.pi + (self.theta - np.pi/2)], [self.theta - np.pi / 2, np.pi]]
        else:
            self.domain_num = 1
            self.domain = [[self.theta - np.pi/2, self.theta + np.pi/2]]


def get_initial_hyperplanes(points):
    new_points = [Point2D(item) for item in points]
    points_used = []
    multi_hyperplanes = {}
    single_hyperplanes = {}
    # 构建一个列表，存储theta角度时可能的点
    for theta in range(-180, 180):
        temp = [[], []]
        num = 0
        for item in new_points:
            dis = item.get_distance(theta * np.pi / 180)
            if dis > -0.1:
                temp[0].append(item)
                temp[1].append(dis)
                num += 1
        # 求均值和方差分类

        multi_hyperplanes[theta] = [num, temp]

    # 处理含平行线的

    # 处理不含平行线的






