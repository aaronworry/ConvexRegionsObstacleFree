from ellipsoidDecomp import EllipsoidDecomp
import numpy as np
from decimal import *
import math



class IterativeDecomp(EllipsoidDecomp):
    def __init__(self, dim, origin):
        super().__init__(dim, origin)

    def dilate_iter(self, path_raw, iterative_num = 5, res = 0, offset = 0):
        path = self.downsample(path_raw, res) if res > 0 else path_raw
        self.dilate(path, offset)
        new_path = self.simplify(path)
        for i in range(iterative_num):
            if len(new_path) == len(path):
                break
            else:
                path = new_path
                self.dilate(path, offset)
                new_path = self.simplify(path)


    def downsample(self, path, res):
        if len(path) < 2:
            return path
        result = []
        for i in range(len(path) - 1):
            dist = Decimal(np.linalg.norm(path[i+1] - path[i]))
            cnt = math.ceil(dist / res)
            for j in range(cnt):
                result.append(path[i-1] + j * (path[i] - path[i - 1]) / cnt)
        result.append(path[-1])
        return result

    def cal_closet_dist(self, point, polygon):
        dist = Decimal(np.inf)
        for item in polygon.hps:
            d = np.abs(item.n.dot(point - item.point))
            if d < dist:
                dist = d
        return dist


    def simplify(self, path):
        if len(path) < 2:
            return path
        new_path = []
        new_path.append(path[0])
        for i in range(len(path) - 2):
            if not (self.polyhedrons_[i+1].inside(path[0]) and self.cal_closet_dist(path[0], self.polyhedrons_[i+1]) > 0.1):
                new_path.append(path[i+1])
        new_path.append(path[-1])
        return new_path

