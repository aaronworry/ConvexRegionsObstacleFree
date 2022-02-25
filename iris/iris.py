import numpy as np
from geometry import Ellipsoid, Polyhedron, Hyperplane
from cvx_ellipsoid import cvx_ellipsoid
from mosek_ellipsoid import inner_ellipsoid
import time

ELLIPSOID_C_EPSILON = 1e-4

class IRISOptions():
    def __init__(self, require_containment=False, required_containment_points=None, error_on_infeasible_start=False, termination_threshold=2e-2, iter_limit=100):
        self.require_containment = require_containment
        self.error_on_infeasible_start = error_on_infeasible_start
        self.termination_threshold = termination_threshold
        self.iter_limit = iter_limit
        self.required_containment_points = required_containment_points

class IRISRegion():
    def __init__(self, problem):
        self.dim = problem.dim
        self.polyhedron = Polyhedron(self.dim, None, None)
        self.ellipsoid = problem.seed

class IRISDebugData():
    def __init__(self):
        self.ellipsoid_history = []
        self.polyhedron_history = []
        self.obstacles = None
        self.bounds = None
        self.iters = None

class IRISProblem():
    def __init__(self, dim):
        self.obstacle_pts = None
        self.bounds = None
        self.dim = dim
        self.seed = None

    def setSeedPoint(self, point, radius = ELLIPSOID_C_EPSILON):
        dim = len(point)
        C = radius * np.eye(dim)
        self.seed = Ellipsoid(dim, C, point)

    def setSeedEllipsoid(self, ellipsoid):
        self.seed = ellipsoid

    def setBounds(self, new_bounds):
        self.bounds = new_bounds

    def addObstacle(self, new_obstacle):
        if not self.obstacle_pts:
            self.obstacle_pts = new_obstacle
        else:
            self.obstacle_pts = np.hstack(self.obstacle_pts, new_obstacle)


def tangent_plane_through_point(ellipsoid, Cinv2, x):
    #Cinv = np.linalg.inv(ellipsoid.C_)
    #Cinv2 = Cinv.dot(Cinv.T)
    temp = 2 * Cinv2.dot(x - ellipsoid.d_)
    nhat = temp / np.linalg.norm(temp)
    return Hyperplane(len(x), nhat, nhat.T.dot(x))


def separating_hyperplanes(obstacle_pts, ellipsoid):
    """

    :param obstacle_pts: shape(dim, n)  n is the number of obstacle point
    :param ellipsoid:
    :return:
    """
    dim = ellipsoid.dim
    n_obs = len(obstacle_pts[0])
    Cinv = np.linalg.pinv(ellipsoid.C_)
    Cinv2 = Cinv.dot(Cinv.T)
    """
    if n_obs == 0:
        polyhedron = Polyhedron(dim, np.zeros((0, dim)), np.zeros)
    """
    planes = []
    img_obs = obstacle_pts.T   # [n, dim]
    img_obs = Cinv.dot(np.transpose(img_obs - ellipsoid.d_))   # [dim, n]
    image_squared = np.linalg.norm(img_obs, axis=0)                # [1, n]
    temp = list(image_squared)
    obs_sort_idx = np.argsort(temp)
    uncovered_obstacles = [True for _ in range(len(temp))]
    for item in obs_sort_idx:
        if not uncovered_obstacles[item]:
            continue
        plane = tangent_plane_through_point(ellipsoid, Cinv2, obstacle_pts[:, item])
        planes.append(plane)

        for j in range(n_obs):
            if uncovered_obstacles[j] and plane.a.T.dot(obstacle_pts[:, j]) - plane.b0 >= 0:
                uncovered_obstacles[j] = False

        uncovered_obstacles[item] = False
        if not any(uncovered_obstacles):
            break

    row = len(planes)
    A = np.zeros((row, dim))
    b = np.zeros(row)

    for i in range(row):
        A[i] = planes[i].a.T
        b[i] = planes[i].b0
    return Polyhedron(dim, A, b)

def inflate_region(problem, options, debug=None):
    ellipsoid = problem.seed
    best_vol = ELLIPSOID_C_EPSILON ** problem.dim
    iter = 0

    if debug:
        debug.bounds = problem.bounds
        debug.ellipsoid_history.append(ellipsoid)
        debug.obstacles = problem.obstacle_pts

    p_time = e_time = 0

    while True:
        # cal the polyhedron
        begin = time.time()
        polyhedron = separating_hyperplanes(problem.obstacle_pts, ellipsoid)
        end = time.time()
        p_time += (end - begin)

        polyhedron.appendConstraints(problem.bounds)

        if options.require_containment:
            if len(options.require_containment_points):
                all_points_contained = True
                for item in options.require_containment_points:
                    if not polyhedron.contains(item, 0.0):
                        all_points_contained = False
                        break
            else:
                all_points_contained = polyhedron.contains(problem.seed.d_, 0.0)

            if all_points_contained:
                if debug:
                    debug.polyhedron_history.append(polyhedron)
            else:
                print("break early because the start point is no longer contained in the polyhedron")
                return polyhedron, ellipsoid

        else:
            if debug:
                debug.polyhedron_history.append(polyhedron)


        # cal the ellipsoid
        # A_ = [list(item) for item in polyhedron.A_]
        # b_ = list(polyhedron.b_)
        begin = time.time()
        # Ci, di = lownerjohn_inner(A_, b_)

        ellipsoid, volume = cvx_ellipsoid(polyhedron.A_, polyhedron.b_)

        # ellipsoid = inner_ellipsoid(polyhedron)
        end = time.time()
        e_time += (end - begin)
        # ellipsoid = Ellipsoid(problem.dim, np.array(Ci), np.array(di))
        # volume = ellipsoid.getVolume()
        if debug:
            debug.ellipsoid_history.append(ellipsoid)

        at_iter_limit = (options.iter_limit > 0) and (iter + 1 >= options.iter_limit)
        insufficient_progress = (abs(volume - best_vol) / best_vol) < options.termination_threshold
        if at_iter_limit or insufficient_progress:
            break

        best_vol = volume
        iter += 1
        if debug:
            debug.iters = iter

    return polyhedron, ellipsoid

