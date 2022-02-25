import mosek
from geometry import Ellipsoid
import numpy as np


def extract_solution(dim, xx, barx, n, ndx_d):
    bar_ndx = 0
    ellipsoid = Ellipsoid(dim, np.zeros((dim, dim)), np.zeros(dim))
    for j in range(2 * n):
        for i in range(j, 2 * n):
            if j < dim and i < dim:
                ellipsoid.setCEntry(i, j, barx[bar_ndx])
                ellipsoid.setCEntry(j, i, barx[bar_ndx])
            bar_ndx += 1
    for i in range(dim):
        ellipsoid.setDEntry(i, xx[ndx_d[i]])
    return ellipsoid


def streamprinter(text):
    print("%s" % text),


## mosek inner_ellipsoid
def inner_ellipsoid(polyhedron):
    m, n = polyhedron.getNumberOfConstraints(), polyhedron.dim
    l = np.ceil(np.log2(n))

    # 1， 2， 1， 2， 10， 5
    num_t, num_d, num_s, num_z, num_f, num_g = 1, n, np.power(2, 1) - 1, np.power(2, 1), m * n, m
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

    ncon = n * m + m + n + n + (np.power(2, 1) - n) + 1 + (n * (n - 1) / 2) + (np.power(2, 1) - 1)
    nabar = n * m * n + n + n + (n * (n - 1) / 2)
    abar_ndx = 0

    env = mosek.Env()
    task = mosek.Task(env, int(ncon), 0)
    task.set_Stream(mosek.streamtype.log, streamprinter)
    task.appendcons(int(ncon))
    task.appendvars(nvar)

    dim_bar = [2 * n]
    len_bar = [n * (n + 1) / 2]

    task.appendbarvars(dim_bar)
    task.putcj(ndx_t[0], 1.0)  # c[ndx_t[0]] = 1.0       c为系数
    for i in range(nvar):
        task.putvarbound(i, mosek.boundkey.fr, -float('inf'), float('inf'))

    bara_i = [0 for _ in range(int(nabar))]
    bara_j = [0 for _ in range(int(nabar))]
    bara_k = [0 for _ in range(int(nabar))]
    bara_l = [0 for _ in range(int(nabar))]
    bara_v = [0 for _ in range(int(nabar))]

    subi_A_row = [0 for _ in range(num_d + 1)]
    vali_A_row = [0. for _ in range(num_d + 1)]

    con_ndx = 0

    """
    A :   m * (n + 1) 行
    constraint :  m * (n + 1) 个
    """
    for i in range(m):  #
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
            subi = [ndx_f[i + m * j]]
            vali = [-1]
            task.putarow(con_ndx, subi, vali)  # A[con_ndx][subi[0]] = vali[0]            n 行  1 列
            task.putconbound(con_ndx, mosek.boundkey.fx, 0, 0)  # constraint[con_ndx] = 0  固定     n 个
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
        task.putarow(con_ndx, subi_A_row, vali_A_row)  # 有多少补充多少        1 行 n 列
        task.putconbound(con_ndx, mosek.boundkey.fx, polyhedron.b_[i],
                         polyhedron.b_[i])  # constraint[con_ndx] = polyhedron.b_[i]  固定   1 个
        """
        MSK_putarow(task, con_ndx, num_d + 1, subi_A_row.data(), vali_A_row.data())
        MSK_putconbound(task, con_ndx, MSK_BK_FX, polyhedron.getB()(i, 0), polyhedron.getB()(i, 0))
        """
        con_ndx += 1

    """
    A :   n 行 1 列
    constraint :  n 个
    """
    for j in range(n):
        # Xbar_[n+j][j] = z_j
        bara_i[abar_ndx] = con_ndx
        bara_j[abar_ndx] = 0
        bara_k[abar_ndx] = n + j
        bara_l[abar_ndx] = j
        bara_v[abar_ndx] = 1
        abar_ndx += 1

        subi = [ndx_z[j]]
        vali = [-1]
        task.putarow(con_ndx, subi, vali)  # n * 1
        task.putconbound(con_ndx, mosek.boundkey.fx, 0, 0)  # n个
        """
        MSK_putarow(task, con_ndx, 1, subi, vali)
        MSK_putconbound(task, con_ndx, MSK_BK_FX, 0, 0)
        """
        con_ndx += 1

    """
    A :   n 行 1 列
    constraint :  n 个
    """
    for j in range(n):
        # Xbar_[n+j][n+j] = z_j
        bara_i[abar_ndx] = con_ndx
        bara_j[abar_ndx] = 0
        bara_k[abar_ndx] = n + j
        bara_l[abar_ndx] = n + j
        bara_v[abar_ndx] = 1
        abar_ndx += 1

        subi = [ndx_z[j]]
        vali = [-1]
        task.putarow(con_ndx, subi, vali)  # n * 1
        task.putconbound(con_ndx, mosek.boundkey.fx, 0, 0)  # n个
        """
        MSK_putarow(task, con_ndx, 1, subi, vali)
        MSK_putconbound(task, con_ndx, MSK_BK_FX, 0, 0)
        """
        con_ndx += 1

    """
    A :   num_z 行 2 列
    constraint :  num_z 个
    """
    for j in range(n, num_z):
        # z_j = t for j > n
        subi = [ndx_z[j], ndx_t[0]]
        vali = [1, -1]
        task.putarow(con_ndx, subi, vali)  # num_z * 2
        task.putconbound(con_ndx, mosek.boundkey.fx, 0, 0)  # num_z
        """
        MSK_putarow(task, con_ndx, 2, subi, vali)
        MSK_putconbound(task, con_ndx, MSK_BK_FX, 0, 0)
        """
        con_ndx += 1

    # Off-diagonal elements of Y22 are 0
    """
    constraint :  n * (n - 1) / 2 个
    A: 这 n * (n - 1) / 2 行 为空
    """
    for k in range(n, 2 * n):
        for l in range(n, k):
            bara_i[abar_ndx] = con_ndx
            bara_j[abar_ndx] = 0
            bara_k[abar_ndx] = k
            bara_l[abar_ndx] = l
            bara_v[abar_ndx] = 1
            abar_ndx += 1
            task.putconbound(con_ndx, mosek.boundkey.fx, 0, 0)  # n * (n - 1) / 2
            """
            MSK_putconbound(task, con_ndx, MSK_BK_FX, 0, 0)
            """
            con_ndx += 1

    assert (abar_ndx == nabar)

    # 2^(l/2)t == s_{2l - 1}
    """
    A: 1行2列
    constraint :  1 个
    """
    subi = [ndx_t[0], ndx_s[num_s - 1]]
    vali = [np.power(2, 1 / 2.), -1]
    task.putarow(con_ndx, subi, vali)  # 1 * 2
    task.putconbound(con_ndx, mosek.boundkey.fx, 0, 0)  # 1
    """
    MSK_putarow(task, con_ndx, 2, subi, vali)
    MSK_putconbound(task, con_ndx, MSK_BK_FX, 0, 0)
    """
    con_ndx += 1

    """
    A: num_s行2列
    constraint :  num_s 个
    """
    for j in range(num_s):
        # s_j = sprime_j
        subi = [ndx_s[j], ndx_sprime[j]]
        vali = [1, -1]
        task.putarow(con_ndx, subi, vali)  # num_s * 2
        task.putconbound(con_ndx, mosek.boundkey.fx, 0, 0)  # num_s
        """
        MSK_putarow(task, con_ndx, 2, subi, vali)
        MSK_putconbound(task, con_ndx, MSK_BK_FX, 0, 0)
        """
        con_ndx += 1

    assert (con_ndx == ncon)

    """
    x: 
    conic constraint : 1
    """
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
        task.appendcone(mosek.conetype.rquad, 0.0, csub)
        """
        MSK_appendcone(task, MSK_CT_RQUAD, 0.0, 3, csub)
        conepar = 0.0, nummem = 3, 

        x = [x[csub[0]], x[csub[1]], x[csub[2]]]

        it means: 2 * x[0] * x[1] >= x[2] * x[2],  x[0] > 0, x[0] > 0
        x = [x[0], x[1], x[2]]
        """
        lhs_idx += 2

    """
    x: 
    conic constraint : m
    """
    csub_f_row = [0 for _ in range(n + 1)]
    for i in range(m):
        csub_f_row[0] = ndx_g[i]
        for j in range(n):
            csub_f_row[j + 1] = ndx_f[i + m * j]
        task.appendcone(mosek.conetype.quad, 0.0, csub_f_row)
        """
        MSK_appendcone(task, MSK_CT_QUAD, 0.0, n + 1, csub_f_row.data())

        it means: x[0] >= sqrt(x[1] ** 2 + ... + x[n] ** 2)
        x = [x[0], x[1], x[2], ..., x[n]]
        """

    # Divide all off-diagonal entries of Abar by 2. This is necessary because Abar
    # is assumed by the solver to be a symmetric matrix, but we're only setting
    # its lower triangular part.
    for i in range(int(nabar)):  # number in A^
        if bara_k[i] != bara_l[i]:
            bara_v[i] /= 2.

    bara_i = [int(item) for item in bara_i]
    bara_j = [int(item) for item in bara_j]
    bara_k = [int(item) for item in bara_k]
    bara_l = [int(item) for item in bara_l]

    task.putbarablocktriplet(int(nabar), bara_i, bara_j, bara_k, bara_l, bara_v)
    task.putobjsense(mosek.objsense.maximize)
    """
    MSK_putbarablocktriplet(task, nabar, bara_i.data(), bara_j.data(), bara_k.data(), bara_l.data(), bara_v.data())
        nabar: Number of elements in the block triplet form
        bara_i: Constraint index
        bara_j: Symmetric matrix variable index
        bara_k: Block row index
        bara_l: Block column index
        bara_v: The numerical value associated with each block triplet

    MSK_putobjsense(task, MSK_OBJECTIVE_SENSE_MAXIMIZE)
    """

    task.optimize()
    task.solutionsummary(mosek.streamtype.msg)
    solsta = task.getsolsta(mosek.soltype.itr)

    ellipsoid = Ellipsoid(polyhedron.dim, np.zeros((polyhedron.dim, polyhedron.dim)), np.zeros(polyhedron.dim))

    if solsta == mosek.solsta.optimal:
        xx = [0.] * nvar
        barx = [0.] * len_bar[0]
        """
        xx = mosek.calloctask(task, nvar, len(mosek.reealt))
        barx = mosek.calloctask(task, len_bar[0], len(mosek.reealt))
        """
        task.getxx(mosek.soltype.itr, xx)  # return x ** x
        task.getbarxj(mosek.soltype.itr, 0, barx)  # x^_0

        ellipsoid = extract_solution(polyhedron.dim, xx, barx, n, ndx_d)

    obj_val = 0
    obj_val = task.getprimalobj(mosek.soltype.itr)
    task.__del__()
    env.__del__()

    return ellipsoid
