from baseDecomp import BaseDecomp
import numpy as np
import os, sys
currentdir = os.path.dirname(os.path.realpath(__file__))
parentdir = os.path.dirname(currentdir)
sys.path.append(parentdir)

from convexRegion.ellipse import Ellipsoid


class SeedDecomp(BaseDecomp):
    def __init__(self, dim, obs, local_bbox, point):
        super().__init__(dim, obs, local_bbox)
        self.seed = point

    def dialate(self, radius):
        self.ellipsoid_ = Ellipsoid(self.dim, radius * np.identity(self.dim), self.seed)
        self.find_polyhedron()
        self.add_local_bbox(self.polyhedron_)

    def add_local_bbox(self, polyhedron):
        # 最初为地图大小
        self.local_bbox_ = polyhedron

if __name__ == "__main__":
    obs = [np.array(1., 1.), np.array(-1., 1.), np.array(-1., -1.), np.array(1., -1.)]
    seed = np.array([0., 0.])
    local_bbox = None

    decomp = SeedDecomp(2, obs, local_bbox, seed)
    decomp.dilate(0.1)