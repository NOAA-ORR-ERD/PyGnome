#!/usr/bin/env python

"""
Code to compute surface surface_concentration from particles

Ultimatley, there may be multiple versions of this
-- with Cython optimizationas and all that.
"""

import numpy as np
from scipy.stats import gaussian_kde


def compute_surface_concentration(sc, algorithm):
    """
    compute the surface concentration from the passed-in spill container

    :param sc: spill container -- data in it wil be usd, and the results will
               be put in a "surface_concentration" array

    :param algorithm: algorithm to use -- currently only "kde" is supported
    """
    if sc['positions'].shape[0] == 0:  # nothing to be done
        return
    if algorithm == 'kde':
        surface_conc_kde(sc)
    else:
        raise ValueError('the only surface concentration algorithm currently supported'
                         'is "kde"')


def surface_conc_kde(sc):
    """
    Computes the surface concentration using scipy's

    Kernel Density Estimator code

    This code does NOT take into account variable mass of the elements!

    We need to fix that!

    a "surface_concentration" array will be added to the spill container

    :param sc: spill container that you want the concentrations computed on
    """
    positions = sc['positions']
    print positions
    if positions.shape[0] > 2:  # can't compute a kde for less than 3 points!
        lon = positions[:, 0]
        lat = positions[:, 1]
        mass = sc['mass']
        print np.unique(lon)
        print np.unique(lat)
        if len(np.unique(lat))>2 or len(np.unique(lon))>2:
            lon0, lat0 = min(lon), min(lat)
            # FIXME: should use projection code to get this right.
            x = (lon - lon0) * 111325 * np.cos(lat0 * np.pi / 180)
            y = (lat - lat0) * 111325
            xy = np.vstack([x, y])
            c = gaussian_kde(xy)(xy)  # units??
            # this is assuming unit mass per point, so we need to scale it
            c *= mass.sum()# / mass.shape[0]
        else:
            c = np.ones((positions.shape[0],))*mass.sum()
    else:
        c = np.zeros((positions.shape[0],))

    sc['surface_concentration'] = c
