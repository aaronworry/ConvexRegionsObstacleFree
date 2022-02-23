import numpy as np
from iris import getGenerators
import math

ELLIPSOID_C_EPSILON = 1e-4

def factorial(n):
    return 1 if n == 0 else factorial(n - 1) * n

def nSphereVolume(dim, radius):
    k = dim // 2
    if dim % 2 == 0:
        v = (math.pi ** k) / factorial(k)
    else:
        v = 2 * factorial(k) * ((4 * math.pi) ** k) / factorial(2 * k + 1)
    return v * (radius ** dim)


class Ellipsoid():
    def __init__(self, dim, C, d):
        self.C_ = C
        self.dim = dim
        self.d_ = d

    def setCEntry(self, row, col, value):
        self.C_[row][col] = value

    def setDEntry(self, idx, value):
        self.d_[idx] = value

    def fromNSphere(self, center, radius):
        dim = len(center)
        C = radius * np.eye(dim)
        return Ellipsoid(dim, C, center)

    def getVolume(self):
        return np.linalg.det(self.C_) * nSphereVolume(self.dim, 1.0)



class Hyperplane():
    def __init__(self, dim, a, b0):
        self.dim = dim
        self.a = a
        self.b0 = b0

class Polyhedron():
    def __init__(self, dim, A, b):
        self.dim = dim
        self.A_ = A
        self.b_ = b
        self.dd_representation_dirty_ = True
        self.generator_points_ = None
        self.generator_rays_ = None

    def updateDDRepresentation(self):
        self.generator_rays_ = None
        self.generator_points_ = None
        getGenerators(self.A_, self.b_, self.generator_points_, self.generator_rays_)
        self.dd_representation_dirty_ = False

    def getNumberOfConstraints(self):
        return len(self.b_)

    def appendConstraints(self, otherPolyhedron):
        self.A_ = np.vstack((self.A_, otherPolyhedron.A_))
        self.b_ = np.vstack((self.b_, otherPolyhedron.b_))
        self.dd_representation_dirty_ = True

    def generatorPoints(self):
        if self.dd_representation_dirty_:
            self.updateDDRepresentation()
        return self.generator_points_

    def generatorRays(self):
        if self.dd_representation_dirty_:
            self.updateDDRepresentation()
        return self.generator_rays_

    def contains(self, point, tolerance):
        return max(self.A_.dot(point) - self.b_) <= tolerance



if __name__ == '__main__':
    print(factorial(5))