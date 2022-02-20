from baseDecomp import BaseDecomp
import numpy as np


class SeedDecomp(BaseDecomp):
    def __init__(self, dim, obs, local_bbox, point):
        super().__init__(dim, obs, local_bbox)
        self.seed = point

    def dialate(self, radius):
        self.ellipsoid_ = Ellipsoid(dim, radius * np.identity(self.dim), self.seed)
        self.find_polyhedron()
        self.add_local_bbox(self.polyhedron_)

    def add_local_bbox(self, polyhedron):
        # 最初为地图大小
        self.local_bbox_ = polyhedron