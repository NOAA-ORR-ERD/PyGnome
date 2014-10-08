'''
Utility functions
'''
from math import exp

import numpy as np

from oil_library.models import KVis


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


def viscosities(oil):
    '''
        Get a list of all kinematic viscosities associated with this
        oil object.  The list is compiled from the stored kinematic
        and dynamic viscosities associated with the oil record.
        The viscosity fields contain:
          - kinematic viscosity in m^2/sec
          - reference temperature in degrees kelvin
          - weathering ???
        Viscosity entries are ordered by (weathering, temperature)
        If we are using dynamic viscosities, we calculate the
        kinematic viscosity from the density that is closest
        to the respective reference temperature
    '''
    # first we get the kinematic viscosities if they exist
    ret = []
    if oil.kvis:
        ret = [(k.m_2_s,
                k.ref_temp_k,
                (0.0 if k.weathering == None else k.weathering))
                for k in oil.kvis
                if k.ref_temp_k != None]

    if oil.dvis:
        # If we have any DVis records, we need to get the
        # dynamic viscosities, convert to kinematic, and
        # add them if possible.
        # We have dvis at a certain (temperature, weathering).
        # We need to get density at the same weathering and
        # the closest temperature in order to calculate the kinematic.
        # There are lots of oil entries where the dvis do not have
        # matching densities for (temp, weathering)
        densities = [(d.kg_m_3,
                      d.ref_temp_k,
                      (0.0 if d.weathering == None else d.weathering))
                     for d in oil.densities]

        for v, t, w in [(d.kg_ms, d.ref_temp_k, d.weathering)
                        for d in oil.dvis]:
            if w == None:
                w = 0.0

            # if we already have a KVis at the same
            # (temperature, weathering), we do not need
            # another one
            if len([vv for vv in ret
                    if vv[1] == t and vv[2] == w]) > 0:
                continue

            # grab the densities with matching weathering
            dlist = [(d[0], abs(t - d[1]))
                     for d in densities
                     if d[2] == w]

            if len(dlist) == 0:
                continue

            # grab the density with the closest temperature
            density = sorted(dlist, key=lambda x: x[1])[0][0]

            # kvis = dvis/density
            ret.append(((v / density), t, w))

    ret.sort(key=lambda x: (x[2], x[1]))
    kwargs = ['m_2_s', 'ref_temp_k', 'weathering']

    # caution: although we will have a list of real
    #          KVis objects, they are not persisted
    #          in the database.
    ret = [(KVis(**dict(zip(kwargs, v)))) for v in ret]
    return ret


def get_viscosity(oil, temp):
    '''
    :param units: optional input if output units should be something other
                  than m^2/s
    :return: Kinematic Viscosity at current temperature.
             Default units: (m^2/s)

    The Oil object has a list of kinematic viscosities at empirically
    measured temperatures.  We need to use the ones closest to our
    current temperature and calculate our viscosity from it.
    '''
    vis = viscosities(oil)
    if vis:
        # first get our v_max
        k_v2 = 5000.0
        pour_point = (oil.pour_point_max_k
                      if oil.pour_point_max_k != None
                      else oil.pour_point_min_k)
        if pour_point:
            try:
                visc = sorted([(v, abs(v.ref_temp_k - pour_point))
                            for v in vis
                            if v != None],
                           key=lambda v: v[1])[0][0]
            except:
                print 'failed on {0.id}, adios_id {0.adios_oil_id}'.format(oil)
            v_ref = visc.m_2_s
            t_ref = visc.ref_temp_k

            v_max = v_ref * exp(k_v2 / pour_point - k_v2 / t_ref)
        else:
            v_max = None

        # now get our v_0
        visc = sorted([(v, abs(v.ref_temp_k - temp)) for v in vis],
                      key=lambda v: v[1])[0][0]
        v_ref = visc.m_2_s
        t_ref = visc.ref_temp_k

        if (temp - t_ref) == 0:
            v_0 = v_ref
        else:
            v_0 = v_ref * exp(k_v2 / temp - k_v2 / t_ref)

        if v_max:
            return (v_max, v_0)[v_0 <= v_max]
        else:
            return v_0
    else:
        return None

# ORIG CODE MOVED FROM OilProps object
#==============================================================================
# def get_density_orig(oil, temp):
#     '''
#     Given an oil object - it will contain a list of density objects with
#     density at a reference temperature. This function computes and returns the
#     density at 'temp'.
# 
#     Function works on list of temps/numpy arrays or a scalar
# 
#     If oil only contains one density value, return that for all temps
# 
#     Function does not do any unit conversion - it expects the data in SI units
#     (kg/m^3) and (K) and returns the output in SI units.
# 
#     Optional 'out' parameter in keeping with numpy convention, fill the out
#     array if provided
# 
#     :return: scalar Density in SI units: (kg/m^3)
#     '''
#     # calculate our density at temperature
#     density_rec = sorted([(d, abs(d.ref_temp_k - temp))
#                           for d in oil.densities],
#                          key=lambda d: d[1])[0][0]
#     d_ref = density_rec.kg_m_3
#     t_ref = density_rec.ref_temp_k
#     k_p1 = 0.008
# 
#     d_0 = d_ref / (1 - k_p1 * (t_ref - temp))
#     return d_0
#
#     def viscosities(self):
#         '''
#             Get a list of all kinematic viscosities associated with this
#             oil object.  The list is compiled from the stored kinematic
#             and dynamic viscosities associated with the oil record.
#             The viscosity fields contain:
#               - kinematic viscosity in m^2/sec
#               - reference temperature in degrees kelvin
#               - weathering ???
#             Viscosity entries are ordered by (weathering, temperature)
#             If we are using dynamic viscosities, we calculate the
#             kinematic viscosity from the density that is closest
#             to the respective reference temperature
#         '''
#         # first we get the kinematic viscosities if they exist
#         ret = []
#         if self._r_oil.kvis:
#             ret = [(k.m_2_s,
#                     k.ref_temp_k,
#                     (0.0 if k.weathering == None else k.weathering))
#                     for k in self._r_oil.kvis
#                     if k.ref_temp_k != None]
# 
#         if self._r_oil.dvis:
#             # If we have any DVis records, we need to get the
#             # dynamic viscosities, convert to kinematic, and
#             # add them if possible.
#             # We have dvis at a certain (temperature, weathering).
#             # We need to get density at the same weathering and
#             # the closest temperature in order to calculate the kinematic.
#             # There are lots of oil entries where the dvis do not have
#             # matching densities for (temp, weathering)
#             densities = [(d.kg_m_3,
#                           d.ref_temp_k,
#                           (0.0 if d.weathering == None else d.weathering))
#                          for d in self._r_oil.densities]
#  
#             for v, t, w in [(d.kg_ms, d.ref_temp_k, d.weathering)
#                             for d in self._r_oil.dvis]:
#                 if w == None:
#                     w = 0.0
#  
#                 # if we already have a KVis at the same
#                 # (temperature, weathering), we do not need
#                 # another one
#                 if len([vv for vv in ret
#                         if vv[1] == t and vv[2] == w]) > 0:
#                     continue
#  
#                 # grab the densities with matching weathering
#                 dlist = [(d[0], abs(t - d[1]))
#                          for d in densities
#                          if d[2] == w]
#  
#                 if len(dlist) == 0:
#                     continue
#  
#                 # grab the density with the closest temperature
#                 density = sorted(dlist, key=lambda x: x[1])[0][0]
#  
#                 # kvis = dvis/density
#                 ret.append(((v / density), t, w))
#  
#         ret.sort(key=lambda x: (x[2], x[1]))
#         kwargs = ['(m^2/s)', 'Ref Temp (K)', 'Weathering']
#  
#         # caution: although we will have a list of real
#         #          KVis objects, they are not persisted
#         #          in the database.
#         ret = [(KVis(**dict(zip(kwargs, v)))) for v in ret]
#         return ret
#  
#     def get_viscosity(self, units='m^2/s'):
#         '''
#         :param units: optional input if output units should be something other
#                       than m^2/s
#         :return: Kinematic Viscosity at current temperature.
#                  Default units: (m^2/s)
#  
#         The Oil object has a list of kinematic viscosities at empirically
#         measured temperatures.  We need to use the ones closest to our
#         current temperature and calculate our viscosity from it.
#         '''
#         if self.viscosities:
#             # first get our v_max
#             k_v2 = 5000.0
#             pour_point = (self._r_oil.pour_point_max_k
#                           if self._r_oil.pour_point_max_k != None
#                           else self._r_oil.pour_point_min_k)
#             if pour_point:
#                 visc = sorted([(v, abs(v.ref_temp_k - pour_point))
#                                 for v in self.viscosities
#                                 if v != None],
#                                key=lambda v: v[1])[0][0]
#                 v_ref = visc.m_2_s
#                 t_ref = visc.ref_temp_k
#  
#                 v_max = v_ref * exp(k_v2 / pour_point - k_v2 / t_ref)
#             else:
#                 v_max = None
#  
#             # now get our v_0
#             visc = sorted([(v, abs(v.ref_temp_k - self.temperature))
#                             for v in self.viscosities],
#                            key=lambda v: v[1])[0][0]
#             v_ref = visc.m_2_s
#             t_ref = visc.ref_temp_k
#  
#             if (self.temperature - t_ref) == 0:
#                 v_0 = v_ref
#             else:
#                 v_0 = v_ref * exp(k_v2 / self.temperature - k_v2 / t_ref)
#  
#             if v_max:
#                 return uc.convert('Kinematic Viscosity', 'm^2/s', units,
#                                   v_0 if v_0 <= v_max else v_max)
#             else:
#                 return uc.convert('Kinematic Viscosity', 'm^2/s', units,
#                                   v_0)
#         else:
#             return None
#==============================================================================

