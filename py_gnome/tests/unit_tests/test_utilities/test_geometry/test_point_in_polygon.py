"""
This code is mostly tested elsewhere:

(though maybe shoudl be copied here)
"""
import numpy as np
from gnome.utilities.geometry.cy_point_in_polygon import points_in_poly

def test_pass_scalar_points_in_poly():
    """
    user found that a scalar array point would get reshaped
    """
    poly = np.array([(0.0, 0.0),
                     (0.0, 5.0),
                     (5.0, 5.0),
                     (5.0, 0.0),
                     ], dtype=np.float64)

    point = np.array([1.0, 2.0, 100.], dtype=np.float64)
    shape = point.shape

    print(shape)
    is_in = points_in_poly(poly, point)

    assert is_in
    assert point.shape == shape


