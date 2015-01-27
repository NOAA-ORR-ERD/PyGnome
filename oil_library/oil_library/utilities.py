'''
Utility functions
'''
from math import exp, log

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
    # make 0-d array into 1-D array
    temp = (temp, temp.reshape(-1))[temp.shape == ()]

    # convert ref_temp and ref_densities to numpy array
    ref_temp = np.asarray([d.ref_temp_k for d in oil.densities], dtype=float)
    d_ref = np.asarray([d.kg_m_3 for d in oil.densities], dtype=float)

    # Change shape to row or column vector for reference temps and densities
    # and also define the axis over which we'll look for argmin()
    if len(temp.shape) == 1 or temp.shape[0] == 1:
        inv_shape = (len(ref_temp), -1)
        axis = 0
    else:
        inv_shape = (-1,)
        ref_temp = ref_temp.reshape(len(ref_temp), -1)
        d_ref = d_ref.reshape(len(d_ref), -1)
        axis = 1

    # Now, use following to create a matrix:
    #    np.abs(temp - ref_temp.reshape(inv_shape))
    #
    # look for argmin in each row (or column) depending on shape of 'temp'
    # This is index where abs(ref_temp[near_idx[ix]] - temp[ix]) is minimum
    near_idx = np.abs(temp - ref_temp.reshape(inv_shape)).argmin(axis)

    k_p1 = 0.008

    if out is None:
        out = np.zeros_like(temp)

    out[:] = d_ref[near_idx] / (1 - k_p1 * (ref_temp[near_idx] - temp))

    return (out, out[0])[len(out) == 1]


def get_v_max(oil, k_v2=5000.0):
    '''
    return oil.pour_point_max_k or oil.pour_point_min_k if pour_point_max_k is
    None.

    todo: ask what is k_v2? It was hard coded in previous implementation
    '''
    # first get our v_max
    pour_point = (oil.pour_point_max_k
                  if oil.pour_point_max_k is not None
                  else oil.pour_point_min_k)
    v_max = None
    if pour_point:
        # Note: All oils seem to have either a pour_point_max_k or
        # pour_point_min_k but leave check in here for now
        visc = sorted([(v, abs(v.ref_temp_k - pour_point))
                       for v in oil.kvis
                       if v is not None],
                      key=lambda v: v[1])[0][0]
        v_ref = visc.m_2_s
        t_ref = visc.ref_temp_k
        v_max = v_ref * exp(k_v2 / t_ref - k_v2 / pour_point)

    return v_max


def get_viscosity(oil, temp, out=None):
    '''
        The Oil object has a list of kinematic viscosities at empirically
        measured temperatures.  We need to use the ones closest to our
        current temperature and calculate our viscosity from it.
        - oil always contains at least one element in the oil.kvis list
    '''
    if not oil.kvis:
        # looks like we have one record - 1255, that does not have kvis
        return None

    k_v2 = 5000.0
    v_max = get_v_max(oil, k_v2)

    # convert to numpy array if it isn't already one
    temp = np.asarray(temp, dtype=float)
    # make 0-d array into 1-D array
    temp = (temp, temp.reshape(-1))[temp.shape == ()]

    # convert ref_temp and ref_densities to numpy array
    ref_temp = np.asarray([v.ref_temp_k for v in oil.kvis], dtype=float)
    v_ref = np.asarray([v.m_2_s for v in oil.kvis], dtype=float)

    # Change shape to row or column vector for reference temps and densities
    # and also define the axis over which we'll look for argmin()
    if len(temp.shape) == 1 or temp.shape[0] == 1:
        inv_shape = (len(ref_temp), -1)
        axis = 0
    else:
        inv_shape = (-1,)
        ref_temp = ref_temp.reshape(len(ref_temp), -1)
        v_ref = v_ref.reshape(len(v_ref), -1)
        axis = 1

    # Now, use following to create a matrix:
    #    np.abs(temp - ref_temp.reshape(inv_shape))
    #
    # look for argmin in each row (or column) depending on shape of 'temp'
    # This is index where abs(ref_temp[near_idx[ix]] - temp[ix]) is minimum
    near_idx = np.abs(temp - ref_temp.reshape(inv_shape)).argmin(axis)

    if out is None:
        out = np.zeros_like(temp)

    # now the actual computation
    out[:] = v_ref[near_idx] * np.exp(k_v2 / temp - k_v2 / ref_temp[near_idx])

    if v_max:
        mask = v_max > out
        if np.any(mask):
            out[mask] = v_max

    return (out, out[0])[len(out) == 1]


def get_boiling_points_from_cuts(oil):
    '''
    Need the mass_fraction to sum up to 1.0
    self.mass_fraction defined as:

        [m0_r, m0_a, m1_r, m1_a, ..., m_resins, m_asphaltenes]

    Also need to understand how to identify saturates/aromatics
    Currently, assumes cuts are added as alternating saturate, then
    aromatic in the list of cuts
    '''
    # distillation cut data available
    mass_fractions = []
    boiling_points = []

    # Note: not sure if we'll get pseudo components from raw cuts so
    # do a temporary update for now
    last_frac = 0.0
    for cut in oil.cuts:
        boiling_points.append(cut.vapor_temp_k)
        mass_fractions.append(round(cut.fraction - last_frac, 4))
        last_frac = cut.fraction

    _add_heavy_component(mass_fractions, boiling_points,
                         get_sara_fraction(oil, 'Resins'))

    _add_heavy_component(mass_fractions, boiling_points,
                         get_sara_fraction(oil, 'Asphaltenes'))

    # add remaining mass to last cut
    # - ask about this, but I think we can assume it is heavy
    _add_heavy_component(mass_fractions, boiling_points, 1.0)

    return zip(boiling_points, mass_fractions)


def get_sara_fraction(oil, sara_type):
    fraction = [f for f in oil.sara_fractions if f.sara_type == sara_type]
    if fraction:
        return fraction[0].fraction
    else:
        return None


def _add_heavy_component(mass_fractions,
                         boiling_points,
                         heavy_comp):
    '''
    Add heavier components to our set of mass fractions.
    - By definition, we need to ensure that the sum of our mass fractions
      is not greater than 1.0
    - Heavier components are assumed to not have a practical
      boiling point
    '''
    if heavy_comp is None or heavy_comp == 0.0:
        return

    f_remain = sum(mass_fractions)
    if f_remain < 1.0:
        if heavy_comp + f_remain <= 1.0:
            mass_fractions.append(heavy_comp)
        else:
            mass_fractions.append(1.0 - f_remain)
        boiling_points.append(float('inf'))


def get_boiling_points_from_api(max_cuts, total_mass, api):
    '''
    return an array of boiling points for each psuedo-components
    Assume max_cuts * 2 components containing [saturate, aromatic]
    Output boiling point in this form:

      components: [s_0, a_0, s_1, a_1, ..., s_i, a_i]
      index, i:   [0, 1, 2, .., max_cuts-1]

    where s_i is boiling point corresponding with i-th saturate component
    similarly a_i is boiling point corresponding with i-th aromatic component
    Hence, max_cuts * 2 components. Boiling point is computed assuming a linear
    relation as follows:

        T_o = 457.16 - 3.3447 * API
        dT/df = 1356.7 - 247.36*ln(API)

    The linear fit is done for evenly spaced intervals and BP is in ascending
    order

        if i % 2 == 0:    # for saturates, i is even
            bp[i] = T_o + dT/df * (i + 1)/(max_cuts * 2)
            so i = [0, 2, 4, .., max_cuts-2]

    The boiling point for i-th component's saturate == aromatic bp:

        if i % 2 == 1:    # aromatic, i is odd
            bp[i] = T_o + dT/df * i/(max_cuts * 2)

    boiling point units of Kelvin
    Boiling point of saturate and aromatic i-th mass component is equal.
    '''
    num_components = max_cuts * 2
    mass_per_component = total_mass / num_components

    saturate_boiling_points = _get_saturate_boiling_points(api, num_components)
    aromatic_boiling_points = _get_aromatic_boiling_points(api, num_components)
    all_boiling_points = [i for sublist in zip(saturate_boiling_points,
                                               aromatic_boiling_points)
                          for i in sublist]

    return zip(all_boiling_points, [mass_per_component] * num_components)


def _get_saturate_boiling_points(api, num_components):
    T_o, dT_dF = _get_saturate_coeffs(api)

    return [dT_dF * (i + 1) / num_components + T_o
            for i in range(0, num_components, 2)]


def _get_saturate_coeffs(api):
    T_o = 457.16 - 3.3447 * api
    dT_dF = 1356.7 - 247.36 * log(api)

    return T_o, dT_dF


def _get_aromatic_boiling_points(api, num_components):
    return _get_saturate_boiling_points(api, num_components)


# ORIG CODE MOVED FROM OilProps object
# ==============================================================================
# def get_density_orig(oil, temp):
#     '''
#     Given an oil object - it will contain a list of density objects with
#     density at a reference temperature. This function computes and returns
#     the density at 'temp'.
#
#     Function works on list of temps/numpy arrays or a scalar
#
#     If oil only contains one density value, return that for all temps
#
#     Function does not do any unit conversion - it expects the data in
#     SI units (kg/m^3) and (K) and returns the output in SI units.
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
#         :param units: optional input if output units should be something
#                       other than m^2/s
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
# ==============================================================================
