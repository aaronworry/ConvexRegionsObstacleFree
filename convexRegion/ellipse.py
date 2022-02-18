import numpy as np
from decimal import Decimal
from polygon import Hyperplane


class Ellipsoid():
    def __init__(self, C, d):
        """
        initial a ellipsoid
        :param C: np.array([[], [], []]) a diagonal matrix, inflate a sphere to a ellipsoid
        :param d: np.array([, ,]) center point of the ellipsoid
        """
        self.C = C
        self.d = d

    def dist(self, pt):
        """
        calculate the relative distance to the center
        :param pt: np.array() a point
        :return: distance of pt and self.d
        """
        return np.linalg.norm(np.linalg.inv(self.C) * (pt - self.d))

    def inside(self, pt):
        """
        judge whether a point is in ellipsoid
        :param pt: a point
        :return: whether pt is in the ellipsoid
        """
        return self.dist(pt) <= 1

    def points_inside(self, pts):
        """
        figure out all point which is in the ellipsoid
        :param pts: [np.array(), np.array()]  a list contain many point
        :return: a list contain point which is inside the ellipsoid
        """
        result = []
        for point in pts:
            if self.inside(point):
                result.append(point)
        return result

    def closest_point(self, pts):
        """
        find the closest point outside the ellipsoid
        :param pts: [np.array(), np.array()]  a list contain many points which are outside the ellipsoid
        :return: the closest point in pts
        """
        min_dist = Decimal(np.inf)
        pt = None
        for point in pts:
            dis = Decimal(self.dist(point))
            if dis < min_dist:
                min_dist = dis
                pt = point
        return pt

    def closest_hyperplane(self, pts):
        """
        find the closest hp
        :param hps: a list include many point
        :return: the closest hp
        """
        closest_pt = self.closest_point(pts)
        n = np.linalg.inv(self.C) * np.linalg.inv(self.C).T * (closest_pt - self.d)
        n_norm = n / np.linalg.norm(n)
        return Hyperplane(closest_pt, n_norm)


    def sample_points_on_ellipsoid_2D(self, num):
        """
        get many point on the contour of ellipsoid
        :param num: the number of sampled point
        :return: a list of points
        """
        delta = Decimal(np.pi*2 / num)
        result = []
        for i in range(num):
            point = self.C * np.array([np.cos(delta*i), np.sin(delta*i)]) + self.d
            result.append(point)
        return result



    def volume(self):
        """
        :return: the volume of ellipsoid
        """
        return np.linalg.det(self.C)

    def print(self):
        print("C=", self.C, "   d=", self.d)



