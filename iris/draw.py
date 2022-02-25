
import matplotlib.pyplot as plt
import scipy.spatial
import numpy as np
from matplotlib.colors import colorConverter
import mpl_toolkits.mplot3d as a3
import matplotlib.animation as animation


def draw(self, ax=None, **kwargs):
    if self.getDimension() == 2:
        return self.draw2d(ax=ax, **kwargs)
    elif self.getDimension() == 3:
        return self.draw3d(ax=ax, **kwargs)
    else:
        raise NotImplementedError("drawing for objects of dimension <2 or >3 not implemented yet")

def draw2d(self, ax=None, **kwargs):
    if ax is None:
        ax = plt.gca()
    points = self.getDrawingVertices()
    kwargs.setdefault("edgecolor", self.default_color)
    return draw_2d_convhull(points, ax, **kwargs)

def draw3d(self, ax=None, **kwargs):
    if ax is None:
        ax = a3.Axes3D(plt.gcf())
    points = self.getDrawingVertices()
    kwargs.setdefault("facecolor", self.default_color)
    return draw_3d_convhull(points, ax, **kwargs)

def draw_convhull(points, ax, **kwargs):
    dim = points.shape[1]
    if dim == 2:
        return draw_2d_convhull(points, ax, **kwargs)
    elif dim == 3:
        return draw_3d_convhull(points, ax, **kwargs)
    else:
        raise NotImplementedError("not implemented for dimension < 2 or > 3")

def draw_2d_convhull(points, ax, **kwargs):
    hull = scipy.spatial.ConvexHull(points)
    kwargs.setdefault("facecolor", "none")
    return [ax.add_patch(plt.Polygon(xy=points[hull.vertices],**kwargs))]

def draw_3d_convhull(points, ax, **kwargs):
    kwargs.setdefault("edgecolor", "k")
    kwargs.setdefault("facecolor", "r")
    kwargs.setdefault("alpha", 0.5)
    kwargs["facecolor"] = colorConverter.to_rgba(kwargs["facecolor"], kwargs["alpha"])
    hull = scipy.spatial.ConvexHull(points)
    artists = []
    for simplex in hull.simplices:
        poly = a3.art3d.Poly3DCollection([points[simplex]], **kwargs)
        if "alpha" in kwargs:
            poly.set_alpha(kwargs["alpha"])
        ax.add_collection3d(poly)
        artists.append(poly)
    return artists

def iterRegions(self):
    return zip(self.polyhedron_history, self.ellipsoid_history)

def animate(self, fig=None, pause=0.5, show=True, repeat_delay=2.0):
    dim = self.bounds.getDimension()
    if dim < 2 or dim > 3:
        raise NotImplementedError("animation is not implemented for dimension < 2 or > 3")
    if fig is None:
        fig = plt.figure()
        if dim == 3:
            ax = a3.Axes3D(fig)
        else:
            ax = plt.gca()

    bounding_pts = np.vstack(self.boundingPoints())
    if bounding_pts.size > 0:
        lb = bounding_pts.min(axis=0)
        ub = bounding_pts.max(axis=0)
        assert(lb.size == dim)
        assert(ub.size == dim)
        width = ub - lb
        ax.set_xlim(lb[0] - 0.1 * width[0], ub[0] + 0.1 * width[0])
        ax.set_ylim(lb[1] - 0.1 * width[1], ub[1] + 0.1 * width[1])
        if dim == 3:
            ax.set_zlim(lb[2] - 0.1 * width[2], ub[2] + 0.1 * width[2])

    artist_sets = []
    for poly, ellipsoid in self.iterRegions():
        artists = []
        d = self.ellipsoid_history[0].getD()
        if dim == 3:
            artists.extend(ax.plot([d[0]], [d[1]], 'go', zs=[d[2]], markersize=10))
        else:
            artists.extend(ax.plot([d[0]], [d[1]], 'go', markersize=10))
        artists.extend(poly.draw(ax))
        artists.extend(ellipsoid.draw(ax))
        for obs in self.getObstacles():
            artists.extend(draw_convhull(obs.T, ax, edgecolor='k', facecolor='k', alpha=0.5))
        artist_sets.append(tuple(artists))

    ani = animation.ArtistAnimation(fig, artist_sets, interval=pause*1000, repeat_delay=repeat_delay*1000)
    if show:
        plt.show()
