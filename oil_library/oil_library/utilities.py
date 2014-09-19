'''
Utility functions
'''

import numpy as np


def get_density(oil, temp, out=None):
    '''
    Given an oil object and temperatures at which density is desired, this
    returns the density at temp. User can provide an array of temps. This
    function will always return a numpy array.

    Following numpy convention, if out is provided, the function writes the
    result into it, and returns a reference to out. out must be the same
    shape as temp
    '''

    # convert to numpy array if it isn't already one
    temp = np.asarray(temp, dtype=float)

    if temp.shape == ():
        # make 0-d array into 1-D array
        temp = temp.reshape(-1)

    # convert ref_temp and ref_densities to numpy array
    ref_temp = [0.] * len(oil.densities)
    d_ref = [0.] * len(oil.densities)
    for ix, d in enumerate(oil.densities):
        ref_temp[ix] = d.ref_temp_k
        d_ref[ix] = d.kg_m_3

    ref_temp = np.asarray(ref_temp, dtype=float)
    d_ref = np.asarray(d_ref, dtype=float)

    # Change shape to row or column vector for reference temps and densities
    # and also define the axis over which we'll look for argmin()
    # For each temp, near_idx is the closest index into ref_temp array where
    # ref_temp is closest to temp
    if len(temp.shape) == 1 or temp.shape[0] == 1:
        inv_shape = (len(ref_temp), -1)
        axis = 0
    else:
        inv_shape = (-1,)
        ref_temp = ref_temp.reshape(len(ref_temp), -1)
        d_ref = d_ref.reshape(len(d_ref), -1)
        axis = 1

    # first find closest matching ref temps to temp
    near_idx = np.abs(temp - ref_temp.reshape(inv_shape)).argmin(axis)

    k_p1 = 0.008

    if out is None:
        out = np.zeros_like(temp)

    out[:] = d_ref[near_idx] / (1 - k_p1 * (ref_temp[near_idx] - temp))

    return (out, out[0])[len(out) == 1]


def get_density_orig(oil, temp):
    '''
    Given an oil object - it will contain a list of density objects with
    density at a reference temperature. This function computes and returns the
    density at 'temp'.

    Function works on list of temps/numpy arrays or a scalar

    If oil only contains one density value, return that for all temps

    Function does not do any unit conversion - it expects the data in SI units
    (kg/m^3) and (K) and returns the output in SI units.

    Optional 'out' parameter in keeping with numpy convention, fill the out
    array if provided

    :return: scalar Density in SI units: (kg/m^3)
    '''
    # calculate our density at temperature
    density_rec = sorted([(d, abs(d.ref_temp_k - temp))
                          for d in oil.densities],
                         key=lambda d: d[1])[0][0]
    d_ref = density_rec.kg_m_3
    t_ref = density_rec.ref_temp_k
    k_p1 = 0.008

    d_0 = d_ref / (1 - k_p1 * (t_ref - temp))
    return d_0
