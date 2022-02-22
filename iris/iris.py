import numpy as np
from geometry import Ellipsoid, Polyhedron, Hyperplane

class IRISOptions():
    def __init__(self, require_containment=False, required_containment_points=None, error_on_infeasible_start=False, termination_threshold=2e-2, iter_limit=100):
        self.require_containment = require_containment
        self.error_on_infeasible_start = error_on_infeasible_start
        self.termination_threshold = termination_threshold
        self.iter_limit = iter_limit
        self.required_containment_points = required_containment_points

class IRISRegion():
    def __init__(self, dim):
        self.dim = dim
        self.polyhedron = Polyhedron(self.dim, None, None)
        self.ellipsoid = Ellipsoid(self.dim, None, None)

class IRISDebugData():
    def __init__(self):
        self.ellipsoid_history = []
        self.polyhedron_history = []
        self.obstacles = None
        self.bounds = None
        self.iters = None

    def boundingPoints(self):
        return self.bounds.generatorPoints()

class IRISProblem():
    def __init__(self, dim):
        self.obstacle_pts = []
        self.bounds = None
        self.dim = dim
        self.seed = None

    def setSeedPoint(self, point):
        self.seed = Ellipsoid.fromNSphere(point, 0.01)

    def setSeedEllipsoid(self, ellipsoid):
        self.seed = ellipsoid

    def setBounds(self, new_bounds):
        self.bounds = new_bounds

    def addObstacle(self, new_obstacle):
        self.obstacle_pts.append(new_obstacle)


def inflate_region(problem, options, debug=None):
    pass


def separating_hyperplanes(obstacle_pts, ellipsoid, polyhedron, infeasible_start):
    pass

# in iris_cdd.h
def getGenerators(A, b, points, rays):
    pass

"""
 in iris_mosek.cpp
"""

def inner_ellipsoid(polyhedron, ellipsoid, existing_env=None):
    pass

def closest_point_in_convex_hull(points, result, existing_env=None):
    pass

def check_res(res):
    pass

def extract_solution():
    pass




