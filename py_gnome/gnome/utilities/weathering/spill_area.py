import numpy as np
from scipy.spatial import ConvexHull
import pyproj

def hull_area(particles):
    """
    :param particles: positions of particles (lon,lat,z)
    :return: area of convex hull in meters square
    """
    hull = ConvexHull(particles[:,0:2])
    pts = hull.points[hull.vertices]
    p = pyproj.Proj(proj='cea',ellps='WGS84', errcheck=True)
    proj_pts = np.array(p(pts[:,0], pts[:,1]))
    x = proj_pts[0]
    y = proj_pts[1]
    return 0.5*np.abs(np.dot(x,np.roll(y,1))-np.dot(y,np.roll(x,1)))