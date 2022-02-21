from lineSegment import LineSegment
import numpy as np

import os, sys
currentdir = os.path.dirname(os.path.realpath(__file__))
parentdir = os.path.dirname(currentdir)
sys.path.append(parentdir)

from convexRegion.polygon import get_linear_constraints

class EllipsoidDecomp():
    def __init__(self, dim, origin):
        self.dim = dim
        self.origin = origin
        self.path_ = None
        self.obs_ = None
        self.ellipsoids_ = None
        self.polyhedrons_ = None  # a list of polyedrons
        self.lines_ = None  # a list of lines

        self.local_bbox_ = None
        self.global_bbox_min_ = None
        self.global_bbox_max_ = None

    def get_constraints(self):
        constraints = [None for _ in range(len(self.polyhedrons_))]
        for i in range(len(self.polyhedrons_)):
            pt = (self.path_[i] + self.path_[i+1]) / 2
            constraints[i] = get_linear_constraints(self.dim, pt, self.polyhedrons_[i].hps)
        return constraints

    def add_global_bbox(self):
        pass

    def dilate(self, path, offset):
        N = len(path) - 1
        self.path_ = path
        self.lines_ = [None for _ in range(N)]
        self.ellipsoid_ = [None for _ in range(N)]
        self.polyhedrons_ = [None for _ in range(N)]

        for i in range(N):
            self.lines_[i] = LineSegment(self.dim, self.obs_, self.local_bbox_, self.path_[i], self.path_[i+1])
            self.lines_[i].dilate(offset)

            self.ellipsoids_[i] = self.lines_[i].ellipsoid_
            self.polyhedrons_[i] = self.lines_[i].pplyhedron_

