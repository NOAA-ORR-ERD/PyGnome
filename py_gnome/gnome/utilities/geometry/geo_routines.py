from shapely.geometry import Polygon
import numpy as np
import random

#tri is a Shapely.Polygon, or 3x2 array of coords
#returns a 2D coordinate
def random_pt_in_tri(tri):
    coords = None
    if isinstance(tri, Polygon):
        coords = tri.exterior.coords
    else:
        coords = tri
    coords = np.array(coords)
    R = random.random()
    S = random.random()
    if R + S >= 1:
        R = 1 - R
        S = 1 - S
    A = coords[0]
    AB = coords[1] - coords[0]
    AC = coords[2] - coords[0]
    RPP = A + R*AB + S*AC
    return RPP