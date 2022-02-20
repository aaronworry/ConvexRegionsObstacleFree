import numpy as np
from decimal import *

epsilon_ = 1e-10

class Hyperplane():
    def __init__(self, dim, point, n):
        """
        initial a hyperplane
        :param point: a point on the boundary of hyperplane
        :param n: a normal vector point to the outside of hyperplane
        """
        self.dim = dim
        self.point = point
        self.n = n

    def signed_dist(self, pt):
        """
        cal the sign-distance between the boundary and point, <0: in hyperplane
        :param pt: a point
        :return: distance
        """
        return Decimal(self.n.dot(pt - self.point))

    def dist(self, pt):
        """
        cal the distance between the boundary and point
        :param pt: a point
        :return: distance
        """
        return abs(self.signed_dist(pt))

class Polygon():
    def __init__(self, dim, hps):
        """
        initial a polygon
        :param hps: a list of hyperplanes
        """
        self.dim = dim
        self.hps = hps

    def inside(self, pt):
        """
        judge whether a point is in polygon
        :param pt: a point
        :return: whether pt is in the polygon
        """
        for hp in self.hps:
            if hp.signed_dist(pt) > epsilon_:
                return False
        return True

    def add(self, hp):
        self.hps.append(hp)

    def points_inside(self, pts):
        """
        figure out all point which is in the polygon
        :param pts: [np.array(), np.array()]  a list contain many point
        :return: a list contain point which is inside the polygon
        """
        result = []
        for point in pts:
            if self.inside(point):
                result.append(point)
        return result

    def cal_normals(self):
        """
        used for visualization
        :return:
        """
        result = []
        for item in self.hps:
            result.append([item.point, item.n])
        return result

    def prunePolygon(self):
        """
        delete surplus hyperplane, calc all intersection
        :return:
        """
        pass

class LinearConstraint():
    def __init__(self, dim, A, b):
        """
        Ax < b
        :param A:
        :param b:
        """
        self.dim = dim
        self.A = A
        self.b = b
        pass

    def inside(self, pt):
        """
        judge whether a point is satisfy the linear constraints
        :param pt: a point
        :return: whether pt is satisfy the linear constraints
        """
        d = self.A * pt - self.b
        for i in range(len(d)):
            if d[i] > 0:
                return False
        return True

def get_linear_constraints(dim, point, hps):
    length = len(hps)
    A = np.zeros((length, len(point)))
    b = np.zeros((length, ))
    for i in range(length):
        n_norm = hps.n
        c = Decimal(hps[i].point.dot(n_norm))
        if n_norm.dot(point) - c > 0:
            n_norm = -n_norm
            c = -c
        A[i] = n_norm
        b[i] = c
    return LinearConstraint(dim, A, b)


