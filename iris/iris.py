import numpy as np
from geometry import Ellipsoid, Polyhedron, Hyperplane
import time
import mosek
import cvxpy as cp

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


def separating_hyperplanes(obstacle_pts, ellipsoid):
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
            if uncovered_obstacles[j] and plane.a.T.dot(obstacle_pts[:, j]) - plane.b0 >= 0:
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
    return Polyhedron(dim, A, b)


def inflate_region(problem, options, debug=None):
    region = IRISRegion(problem.dim)
    best_vol = ELLIPSOID_C_EPSILON ** problem.dim
    volume = 0
    iter = 0
    new_poly = Polyhedron(problem.dim, None, None)

    if debug:
        debug.bounds = problem.bounds
        debug.ellipsoid_history.append(region.ellipsoid)
        obstacles = problem.obstacle_pts

    p_time = e_time = 0

    while True:
        # cal the polyhedron
        begin = time.time()
        new_poly = separating_hyperplanes(problem.obstacle_pts, region.ellipsoid)
        end = time.time()
        p_time += (end - begin)

        new_poly.appendConstraints(problem.bounds)

        if options.require_containment:
            if len(options.require_containment_points):
                all_points_contained = True
                for item in options.require_containment_points:
                    if not new_poly.contains(item, 0.0):
                        all_points_contained = False
                        break
            else:
                all_points_contained = new_poly.contains(problem.seed.d_, 0.0)

            if all_points_contained:
                region.polyhedron = new_poly
                if debug:
                    debug.polyhedron_histroy.append(new_poly)
            else:
                print("break early because the start point is no longer contained in the polyhedron")
                return region

        else:
            region.polyhedron = new_poly
            if debug:
                debug.polyhedron_histroy.append(new_poly)


        # cal the ellipsoid
        begin = time.time()
        volume = inner_ellipsoid(region.polyhedron, region.ellipsoid)
        end = time.time()
        e_time += (end - begin)

        if debug:
            debug.ellipsoid_history.append(region.ellipsoid)

        at_iter_limit = (options.iter_limit > 0) and (iter + 1 >= options.iter_limit)
        insufficient_progress = (abs(volume - best_vol) / best_vol) < options.termination_threshold
        if at_iter_limit or insufficient_progress:
            break

        best_vol = volume
        iter += 1
        if debug:
            debug.iters = iter

    return region


"""
 in iris_mosek.cpp
 
 pip install mosek, cvxpy
"""

def extract_solution(xx, barx, n, ndx_d, ellipsoid):
    bar_ndx = 0
    result = ellipsoid
    for j in range(2*n):
        for i in range(j,2*n):
            if j < ellipsoid.dim and i < ellipsoid.dim:
                result.setCEntry(i, j, barx[bar_ndx])
                result.setCEntry(j, i, barx[bar_ndx])
            barx += 1
    for i in range(ellipsoid.dim):
        result.setDEntry(i, xx[ndx_d[i]])


def inner_ellipsoid(polyhedron, ellipsoid):
    m, n, = polyhedron.getNumberOfConstraints(), polyhedron.dim
    l = np.ceil(np.log2(n))

    # 1， 2， 1， 2， 10， 5
    num_t, num_d, num_s, num_z, num_f, num_g = 1, n, np.pow(2, 1) - 1, np.pow(2, 1), m * n, m
    num_sprime, nvar = num_s, 0


    ndx_t = [nvar + i for i in range(num_t)]
    nvar += len(ndx_t)
    ndx_d = [nvar + i for i in range(num_d)]
    nvar += len(ndx_d)
    ndx_s = [nvar + i for i in range(num_s)]
    nvar += len(ndx_s)
    ndx_sprime = [nvar + i for i in range(num_sprime)]
    nvar += len(ndx_sprime)
    ndx_z = [nvar + i for i in range(num_z)]
    nvar += len(ndx_z)
    ndx_f = [nvar + i for i in range(num_f)]
    nvar += len(ndx_f)
    ndx_g = [nvar + i for i in range(num_g)]
    nvar += len(ndx_g)

    ncon = n * m + m + n + n + (np.pow(2, 1) - n) + 1 + (n * (n - 1) / 2) + (np.pow(2, 1) - 1)
    nabar = n * m * n + n + n + (n * (n - 1) / 2)
    abar_ndx = 0

    """
    MSK_maketask(*env, ncon, 0, &task)
    MSK_linkfunctotaskstream(task,MSK_STREAM_LOG,NULL,printstr)
    MSK_appendcons(task, ncon)
    MSK_appendvars(task, nvar)
    """

    dim_bar = [None for _ in range(2 * n)]
    len_bar = [None for _ in range(n * (n + 1) / 2)]

    """
    MSK_appendbarvars(task, 1, dim_bar)
    MSK_putcj(task, ndx_t[0], 1.0)
    for i in range(nvar):
        MSK_putvarbound(task, i, MSK_BK_FR, -MSK_INFINITY, MSK_INFINITY)
    """

    bara_i = [0 for _ in range(nabar)]
    bara_j = [0 for _ in range(nabar)]
    bara_k = [0 for _ in range(nabar)]
    bara_l = [0 for _ in range(nabar)]
    bara_v = [0 for _ in range(nabar)]

    subi_A_row = [0 for _ in range(num_d + 1)]
    vali_A_row = [0. for _ in range(num_d + 1)]

    con_ndx = 0

    for i in range(m):
        # a_i.T C = [f_[i][1], ..., f_[i][n]]
        for j in range(n):
            # a_i.T C _ j = f_[i][j]
            for k in range(n):
                bara_i[abar_ndx + k] = con_ndx
                bara_j[abar_ndx + k] = 0
                if j >= k:
                    bara_k[abar_ndx + k] = j
                    bara_l[abar_ndx + k] = k
                else:
                    bara_k[abar_ndx + k] = k
                    bara_l[abar_ndx + k] = j
                bara_v[abar_ndx + k] = polyhedron.A_[i][k]
            abar_ndx += n
            subi = [None for _ in range(ndx_f[i + m * j])]
            vali = [None for _ in range(-1)]
            """
            MSK_putarow(task, con_ndx, 1, subi, vali)
            MSK_putconbound(task, con_ndx, MSK_BK_FX, 0, 0)
            """
            con_ndx += 1
        for j in range(num_d):
            subi_A_row[j] = ndx_d[j]
            vali_A_row[j] = 1
        subi_A_row[num_d] = ndx_g[i]
        vali_A_row[num_d] = 1
        """
        MSK_putarow(task, con_ndx, num_d + 1, subi_A_row.data(), vali_A_row.data())
        MSK_putconbound(task, con_ndx, MSK_BK_FX, polyhedron.getB()(i, 0), polyhedron.getB()(i, 0))
        """
        con_ndx += 1

    for j in range(n):
        # Xbar_[n+j][j] = z_j
        bara_i[abar_ndx] = con_ndx
        bara_j[abar_ndx] = 0
        bara_k[abar_ndx] = n + j
        bara_l[abar_ndx] = j
        bara_v[abar_ndx] = 1
        abar_ndx += 1

        subi = [None for _ in range(ndx_z[j])]
        vali = [None for _ in range(-1)]
        """
        MSK_putarow(task, con_ndx, 1, subi, vali)
        MSK_putconbound(task, con_ndx, MSK_BK_FX, 0, 0)
        """
        con_ndx += 1

    for j in range(n):
        # Xbar_[n+j][n+j] = z_j
        bara_i[abar_ndx] = con_ndx
        bara_j[abar_ndx] = 0
        bara_k[abar_ndx] = n + j
        bara_l[abar_ndx] = n + j
        bara_v[abar_ndx] = 1
        abar_ndx += 1

        subi = [None for _ in range(ndx_z[j])]
        vali = [None for _ in range(-1)]
        """
        MSK_putarow(task, con_ndx, 1, subi, vali)
        MSK_putconbound(task, con_ndx, MSK_BK_FX, 0, 0)
        """
        con_ndx += 1

    for j in range(n, num_z):
        # z_j = t for j > n
        subi = [None for _ in range(ndx_z[j], ndx_t[0])]
        vali = [None for _ in range(1, -1)]
        """
        MSK_putarow(task, con_ndx, 2, subi, vali)
        MSK_putconbound(task, con_ndx, MSK_BK_FX, 0, 0)
        """
        con_ndx += 1

    # Off-diagonal elements of Y22 are 0
    for k in range(n, 2*n):
        for l in range(n, k):
            bara_i[abar_ndx] = con_ndx
            bara_j[abar_ndx] = 0
            bara_k[abar_ndx] = k
            bara_l[abar_ndx] = l
            bara_v[abar_ndx] = 1
            abar_ndx += 1
            """
            MSK_putconbound(task, con_ndx, MSK_BK_FX, 0, 0)
            """
            con_ndx += 1

    assert(abar_ndx==nabar)

    # 2^(l/2)t == s_{2l - 1}
    subi = [None for _ in range(ndx_t[0], ndx_s[num_s - 1])]
    vali = [None for _ in range(np.pow(2, 1/2.), -1)]
    """
    MSK_putarow(task, con_ndx, 2, subi, vali)
    MSK_putconbound(task, con_ndx, MSK_BK_FX, 0, 0)
    """
    con_ndx += 1

    for j in range(num_s):
        # s_j = sprime_j
        subi = [None for _ in range(ndx_s[j], ndx_sprime[j])]
        vali = [None for _ in range(1, -1)]
        """
        MSK_putarow(task, con_ndx, 2, subi, vali)
        MSK_putconbound(task, con_ndx, MSK_BK_FX, 0, 0)
        """
        con_ndx += 1

    assert (con_ndx == ncon)

    csub = [0 for _ in range(3)]
    lhs_idx = 0
    for j in range(num_s):
        if lhs_idx < num_z:
            csub[0] = ndx_z[lhs_idx]
        else:
            csub[0] = ndx_sprime[lhs_idx - num_z]

        if lhs_idx + 1 < num_z:
            csub[1] = ndx_z[lhs_idx + 1]
        else:
            csub[1] = ndx_sprime[lhs_idx + 1 - num_z]

        csub[2] = ndx_s[j]
        """
        MSK_appendcone(task, MSK_CT_RQUAD, 0.0, 3, csub)
        """
        lhs_idx += 2

    csub_f_row = [0 for _ in range(n + 1)]
    for i in range(m):
        csub_f_row[0] = ndx_g[i]
        for j in range(n):
            csub_f_row[j + 1] = ndx_f[i + m * j]
        """
        MSK_appendcone(task, MSK_CT_QUAD, 0.0, n + 1, csub_f_row.data())
        """

    # Divide all off-diagonal entries of Abar by 2. This is necessary because Abar
    # is assumed by the solver to be a symmetric matrix, but we're only setting
    # its lower triangular part.
    for i in range(nabar):
        if bara_k[i] != bara_l[i]:
            bara_v[i] /= 2

    """
    MSK_putbarablocktriplet(task, nabar, bara_i.data(), bara_j.data(), bara_k.data(), bara_l.data(), bara_v.data())
    MSK_putobjsense(task, MSK_OBJECTIVE_SENSE_MAXIMIZE)
    """

    xx = None
    barx = None
    trmcode = None
    res = mosek.optimizetrm(task, trmcode)
    mosek.solutionsummary(task, mosek.MSK_STream_MSG)

    solsta = mosek.getsolsta(task, MSK_SOL_ITR)

    if solsta == MSK_SOL_STA_NEAR_OPTIMAL:
        xx = mosek.calloctask(task, nvar, len(mosek.reealt))
        barx = mosek.calloctask(task, len_bar[0], len(mosek.reealt))
        xx = mosek.getxx(task, MSK_SOL_ITR, xx)
        barx = mosek.getbarxj(task, MSK_SOL_ITR, 0, barx)

        extract_solution(xx, barx, n, ndx_d, ellipsoid)

    obj_val = 0
    obj_val = mosek.getprimalobj(task, MSK_SOL_ITR, obj_val)

    return ellipsoid.getVolume()

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

