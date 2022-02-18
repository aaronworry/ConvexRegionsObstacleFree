import numpy as np
from decimal import *

def transform_vec(t, tf):
    """
    transforming a vector
    :param t: a list of vector
    :param tf: a matrix
    :return: transform t
    """
    result = []
    for item in t:
        result.append(tf*item)
    return result

def total_distance(vs):
    """
    cal distance
    :param vs: a list of vector
    :return: total distance
    """
    dist = Decimal(0)
    for i in range(len(vs)-1):
        dist += np.linalg.norm(vs[i+1] - vs[i])
    return dist