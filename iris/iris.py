import numpy as np
from geometry import Ellipsoid, Polyhedron, Hyperplane

ELLIPSOID_C_EPSILON = 1e-4

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


def tangent_plane_through_point(ellipsoid, Cinv2, x):
    #Cinv = np.linalg.inv(ellipsoid.C_)
    #Cinv2 = Cinv.dot(Cinv.T)
    temp = 2 * Cinv2.dot(x - ellipsoid.d_)
    nhat = temp / np.linalg.norm(temp)
    return Hyperplane(len(x), nhat, nhat.T.dot(x))


def separating_hyperplanes(obstacle_pts, ellipsoid, polyhedron):
    dim = ellipsoid.dim
    n_obs = len(obstacle_pts[0])
    Cinv = np.linalg.inv(ellipsoid.C_)
    Cinv2 = Cinv.dot(Cinv.T)
    """
    if n_obs == 0:
        polyhedron = Polyhedron(dim, np.zeros((0, dim)), np.zeros)
    """
    planes = []
    img_obs = obstacle_pts.T
    img_obs = Cinv.dot(np.transpose(img_obs - ellipsoid.d_))   # [dim, n]
    image_squared = np.linalg.norm(img_obs, axis=0)                # [1, n]
    temp = list(image_squared)
    obs_sort_idx = np.argsort(temp)
    uncovered_obstacles = [True for _ in np.traspose(obstacle_pts)]
    for item in obs_sort_idx:
        if not uncovered_obstacles[item]:
            continue
        plane = tangent_plane_through_point(ellipsoid, Cinv2, obstacle_pts[:, item])
        planes.append(plane)

        for j in range(n_obs):
            if uncovered_obstacles[j] and plane.a.T.dot(obstacle_pts[:, j]) - plane.b0 >= -1 * ELLIPSOID_C_EPSILON:
                uncovered_obstacles[j] = False

        uncovered_obstacles[item] = False
        if not any(uncovered_obstacles):
            break

    row = len(planes)
    A = np.zeros((row, dim))
    b = np.array([0 for _ in range(row)])

    for i in range(row):
        A[i] = planes[i].a.T
        b[i] = planes[i].b0
    polyhedron.A_ = A
    polyhedron.b_ = b


def inflate_region(problem, options, debug=None):
    pass




# in iris_cdd.h
def getGenerators(A, b, points, rays):
    # need ???????????????????
    dim = len(A[0])
    row = len(A)
    hrep = np.zeros((row, dim + 1))   # [b -A]
    for i in range(row):
        hrep[i][0] = b[i]
        for j in range(dim):
            hrep[i][j + 1] = -1 * A[i][j]

    generators = dd_DDMatrix2Poly(hrep)

    for i in range(row):
        point_or_ray = np.array([0 for _ in range(dim)])
        for j in range(dim):
            point_or_ray[j] = generators[i][j + 1]
        if generators[i][0] == 0:
            rays.append(point_or_ray)
        else:
            points.append(point_or_ray)




"""
 in iris_mosek.cpp
"""

def check_res(res):
    pass

def extract_solution():
    pass


def inner_ellipsoid(polyhedron, ellipsoid, existing_env=None):
    pass

"""
others
"""
def dd_DDMatrix2Poly(M):
    poly = np.zeros((len(M), len(M[0])))
    for i in range(len(M)):
        for j in range(len(M[0])):
            poly[i][j] = M[i][j]
    return poly # ?????

def arg_sort(vector):
    return [idx for idx, value in sorted(enumerate(vector), key = lambda x: x[1])]
    # return np.argsort(vector)



