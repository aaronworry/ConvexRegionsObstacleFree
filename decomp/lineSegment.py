import numpy as np
from baseDecomp import BaseDecomp
from decimal import *


epsilon_ = 1e-10
class Line():
    def __init__(self, dim, p1, p2):
        self.dim = dim
        self.p1 = p1
        self.p2 = p2

class LineSegment(BaseDecomp):
    def __init__(self, dim, obs, local_bbox, p1, p2):
        super().__init__(dim, obs, local_bbox)
        self.p1 = p1
        self.p2 = p2


    def dilate(self, radius):
        self.find_ellipsoid_2D(radius)
        self.find_polyhedron()
        self.add_local_bbox(self.polyhedron_)

    def get_line_segment(self):
        return Line(self.dim, self.p1, self.p2)

    def add_local_bbox(self, polyhedron):
        # 最初为地图大小
        self.local_bbox_ = polyhedron

    def find_ellipsoid_2D(self, offset):
        f = Decimal(np.linalg.norm(self.p1-self.p2) / 2)
        C = f * np.identity(2)
        axes = np.array([f, f])

        C[0][0] += offset
        axes[0] += offset

        if axes[0] > 0:
            ratio = axes[1] / axes[0]
            axes *= ratio
            C *= ratio

        Ri = vec2_to_rotation(self.p2 - self.p1)
        C = Ri.dot(C).dot(Ri.T)

        ellipsoid = Ellipsoid(2, C, (self.p1 + self.p2)/2)

        obs = ellipsoid.points_inside(self.obs_)
        obs_inside = obs
        while len(obs_inside) > 0:
            pw = ellipsoid.closest_point(obs_inside)
            p = Ri.T.dot(pw - ellipsoid.d)
            if p[0] < axes[0]:
                axes[1]= np.abs(p[1]) / np.sqrt(1 - (p[0] / axes[0])**2)
            new_C = np.identity(2)
            new_C[0][0] = axes[0]
            new_C[1][1] = axes[1]
            ellipsoid.C = Ri.dot(new_C).dot(Ri.T)

            obs_new = []
            for item in obs_inside:
                if 1 - ellipsoid.dist(item) > epsilon_:
                    obs_new.append(item)
            obs_inside = obs_new

        self.ellipsoid_ = ellipsoid

    def find_ellipsoid_3D(self, offset):
        f = Decimal(np.linalg.norm(self.p1 - self.p2) / 2)
        C = f * np.identity(3)
        axes = np.array([f, f, f])

        C[0][0] += offset
        axes[0] += offset

        if axes[0] > 0:
            ratio = axes[1] / axes[0]
            axes *= ratio
            C *= ratio

        Ri = vec3_to_rotation(self.p2 - self.p1)
        C = Ri.dot(C).dot(Ri.T)

        ellipsoid = Ellipsoid(3, C, (self.p1 + self.p2) / 2)
        Rf = Ri

        obs = ellipsoid.points_inside(self.obs_)
        obs_inside = obs
        while len(obs_inside) > 0:
            pw = ellipsoid.closest_point(obs_inside)
            p = Ri.T.dot(pw - ellipsoid.d)
            roll = np.atan2(p[2], p[1])
            Rf = Ri * Quatf(np.cos(roll/2), np.sin(roll/2), 0, 0)
            p = Rf.T.dot(pw - ellipsoid.d)

            if p[0] < axes[0]:
                axes[1] = np.abs(p[1]) / np.sqrt(1 - (p[0] / axes[0]) ** 2)
            new_C = np.identity(3)
            new_C[0][0] = axes[0]
            new_C[1][1] = axes[1]
            new_C[2][2] = axes[2]
            ellipsoid.C = Rf.dot(new_C).dot(Rf.T)

            obs_new = []
            for item in obs_inside:
                if 1 - ellipsoid.dist(item) > epsilon_:
                    obs_new.append(item)
            obs_inside = obs_new

        C = f * np.identity(3)
        C[0][0] = axes[0]
        C[1][1] = axes[1]
        C[2][2] = axes[2]
        ellipsoid.C = Rf.dot(C).dot(Rf.T)
        obs_inside = ellipsoid.points_inside(obs)

        while len(obs_inside) > 0:
            pw = ellipsoid.closest_point(obs_inside)
            p = Rf.T.dot(pw - ellipsoid.d)
            dd = 1 - (p[0] / axes[0]) ** 2 - (p[1] / axes[1]) ** 2
            if dd > epsilon_:
                axes[2] = np.abs(p[2]) / np.sqrt(dd)
            new_C = np.identity(3)
            new_C[0][0] = axes[0]
            new_C[1][1] = axes[1]
            new_C[2][2] = axes[2]
            ellipsoid.C = Rf.dot(new_C).dot(Rf.T)

            obs_new = []
            for item in obs_inside:
                if 1 - ellipsoid.dist(item) > epsilon_:
                    obs_new.append(item)
            obs_inside = obs_new

        self.ellipsoid_ = ellipsoid
