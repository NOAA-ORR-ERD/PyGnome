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
    if sc['positions'].shape[0] == 0 or not algorithm:  # nothing to be done
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

   

    a "surface_concentration" array will be added to the spill container

    :param sc: spill container that you want the concentrations computed on
    """
    spill_num = sc['spill_num']
    sc['surface_concentration'] = np.zeros(spill_num.shape[0],)
    for s in np.unique(spill_num):
        sid = np.where(spill_num==s)
        positions = sc['positions'][sid]
        mass = sc['mass'][sid]
        age = sc['age'][sid]
        c = np.zeros(positions.shape[0],)
        lon = positions[:, 0]
        lat = positions[:, 1]

        bin_length = 6*3600 #kde will be calculated on particles 0-6hrs, 6-12hrs,...
        t = age.min()
        max_age = age.max()
        
        while t<=max_age:
            id = np.where((age<t+bin_length))[0] #we use all particles < t + bin_length for kernel
            lon_for_kernel = lon[id]
            lat_for_kernel = lat[id]
            age_for_kernel = age[id]
            mass_for_kernel = mass[id]
            id_bin = np.where(age_for_kernel>=t)[0] #we only calculate pdf for particles in bin

            if len(np.unique(lat_for_kernel))>2 or len(np.unique(lon_for_kernel))>2: # can't compute a kde for less than 3 unique points!
                lon0, lat0 = min(lon_for_kernel), min(lat_for_kernel)
                # FIXME: should use projection code to get this right.
                x = (lon_for_kernel - lon0) * 111325 * np.cos(lat0 * np.pi / 180)
                y = (lat_for_kernel - lat0) * 111325
                xy = np.vstack([x, y])
                if len(np.unique(mass_for_kernel)) > 1:
                    kernel = gaussian_kde(xy,weights=mass_for_kernel/mass_for_kernel.sum()) 
                else:
                    kernel = gaussian_kde(xy) 
                if mass_for_kernel.sum() > 0:
                    c[id[id_bin]] = kernel(xy[:,id_bin]) * mass_for_kernel.sum() 
                else:
                    c[id[id_bin]] = kernel(xy[:,id_bin]) * len(mass_for_kernel)
            t = t + bin_length
            
        sc['surface_concentration'][sid] = c
