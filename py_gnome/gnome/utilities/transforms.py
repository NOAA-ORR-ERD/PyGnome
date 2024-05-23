
import numpy as np


def r_theta_to_uv_current(r_theta):
    """
    Converts array of current values given with magnitude, direction into
    (u,v) wind values. Current from 0 deg is (u,v) = (0,1), from 45 deg (u,v) =
    (1,1) i.e. rotate clockwise from North. In addition, (u,v) represents
    the direction the current moves towards.

    :param r_theta: NX2 numpy array containing r = r_theta[:,0],
                    theta = r_theta[:,1]. Theta is in degrees between
                    0 and 360.

    :returns: NX2 numpy array containing the corresponding uv Cartesian
              velocity vector
    """

    xform = np.array([(-1., 0.), (0., -1.)])
    return np.dot(r_theta_to_uv_wind(r_theta), xform)


def uv_to_r_theta_current(uv):
    """
    Converts array of current values given with (u,v) current values to
    magnitude, direction.
    Current from 0 deg is (u,v) = (0,1), from 45 deg (u,v) =
    (1,1) i.e. rotate clockwise from North. In addition, (u,v) represents
    the direction the current blows towards

    :param uv: NX2 numpy array, where each row corresponds with a velocity
               vector
    :returns: NX2 numpy array containing polar coordinates r_theta
    """

    xform = np.array([(-1., 0.), (0., -1.)])
    uv = np.dot(uv.reshape(-1, 2), xform)
    return uv_to_r_theta_wind(uv)


def r_theta_to_uv_wind(r_theta):
    """
    Converts array of wind values given with magnitude, direction into(u,v)
    wind values. Wind from 0 deg is (u,v) = (0,-1), from 45 deg (u,v) =
    (-1,-1) i.e. rotate clockwise from North. In addition,
    (u,v) represents the direction the wind blows towards

    :param r_theta: NX2 numpy array containing r = r_theta[:,0],
                    theta = r_theta[:,1]. Theta is in degrees between
                    0 and 360.
    :returns: NX2 numpy array containing the corresponding uv Cartesian
              velocity vector
    """
    r_theta = np.asarray(r_theta, dtype=np.float64).reshape(-1, 2)
    if np.any(r_theta[:, 1] > 360) or np.any(r_theta[:, 1] < 0):
        raise ValueError('input angle in r_theta[:,1] must be '
                         'between 0 and 360')

    if np.any(r_theta[:, 0] < 0):
        raise ValueError('input magnitude in r_theta[:,0] must be '
                         'greater than, equal to 0')

    rq = np.array(r_theta)
    rq[:, 1] = np.deg2rad(rq[:, 1])

    uv = np.zeros_like(rq)
    uv[:, 0] = np.round(rq[:, 0] * np.sin(rq[:, 1]), decimals=14)
    uv[:, 1] = np.round(rq[:, 0] * np.cos(rq[:, 1]), decimals=14)

    # create matrix so -1*0 = 0 and not -0 and let's not screw up original

    uv = -1 * uv

    return uv


def uv_to_r_theta_wind(uv):
    """
    Converts array of wind values given with (u,v) wind values to magnitude,
    direction. Wind from 0 deg is (u,v) = (0,-1), from 45 deg (u,v) =
    (-1,-1) i.e. rotate clockwise from North. In addition,
    (u,v) represents the direction the wind blows towards

    :param uv: NX2 numpy array, where each row corresponds with a velocity
               vector
    :returns: NX2 numpy array containing polar coordinates r_theta
    """

    uv = np.asarray(uv, dtype=np.float64).reshape(-1, 2)
    r_theta = np.zeros_like(uv)
    r_theta[:, 0] = np.apply_along_axis(np.linalg.norm, 1, uv)

    # NOTE: Since desired angle is different from the angle that
    #       arctan2 outputs; the uv array is transformed (multiply by -1)
    #       and atan2 is called with (u,v)
    #       Only to ensure we get the angle per the Wind convention

    uv = -1 * uv  # create new uv object
    r_theta[:, 1] = (np.rad2deg(np.arctan2(uv[:, 0], uv[:, 1])) + 360) % 360

    return r_theta
