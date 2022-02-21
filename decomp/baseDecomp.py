import os, sys
currentdir = os.path.dirname(os.path.realpath(__file__))
parentdir = os.path.dirname(currentdir)
sys.path.append(parentdir)

from convexRegion.polygon import Polygon


class BaseDecomp():
    def __init__(self, dim, obs, local_bbox):
        self.dim = dim
        self.obs_ = obs                        # a list of points
        self.local_bbox_ = local_bbox          # a Polygon
        self.ellipsoid_ = None                 # a ellipse
        self.polyhedron_ = None                # a Polygon



    def set_local_bbox(self, bbox):
        self.lobal_bbox_ = bbox

    def set_obs(self, obs):
        self.obs_ = self.local_bbox_.points_inside(obs)

    def get_ellipsoid(self):
        return self.ellipsoid_

    def get_polyhedron(self):
        return self.polyhedron_

    def dilate(self, radius):
        pass

    def shrink(self, shrink_distance):
        pass

    def add_local_bbox(self, polyhedron):
        pass

    def find_polyhedron(self):
        obs_remain = self.obs_
        polyhedron = []
        while len(obs_remain) > 0:
            v = self.ellipsoid_.closest_hyperplane(obs_remain)
            polyhedron.append(v)
            obs_temp = []
            for item in obs_remain:
                if v.signed_dist(item) < 0:
                    obs_temp.append(item)
            obs_remain = obs_temp
        self.polyhedron_ = Polygon(self.dim, polyhedron)

