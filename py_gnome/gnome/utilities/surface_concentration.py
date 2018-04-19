#!/usr/bin/env python

"""
Code to compute surface surface_concentration from particles

Ultimatley, there may be multiupel versions of this
-- with Cython optimizationas and all that.
"""

def surface_conc_kde(sc):
    """
    Computes the surface concentration using scipy's

    Kernal Density Estimator code

    This code does NOT take into account variable mass of the elements!


    :param sc: spill container that you want the concentations computed on
    """

# # calculate and add particle density data

#               if 'pdf' not in rg_vars: #this part should be moved to initialization but has to be added to spill container
#                   rootgrp.createVariable('pdf','d',('data',))

        if step_num == 0:
            rg_vars['pdf'][self._start_idx:_end_idx] = 0
        else:
            lon = sc['positions'][:, 0]
            lat = sc['positions'][:, 1]
            lon0, lat0 = min(lon), min(lat)
            x = (lon - lon0) * 111325 * np.cos(lat0 * np.pi / 180)
            y = (lat - lat0) * 111325
            xy = np.vstack([x,y])
            z = gaussian_kde(xy)(xy)  # units??

            rg_vars['pdf'][self._start_idx:_end_idx] = z