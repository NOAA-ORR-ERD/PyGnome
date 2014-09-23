'''
added this module for constructing an Oil object from a dict
It could be duck typed, though right now it defines a models.Oil object
Currently only works/tested for _sample_oils dict which only contains name
and API - needs work, but will suffice for testing
'''
from math import exp, log   # for scalars, python is faster

from hazpy import unit_conversion
uc = unit_conversion

from .models import Oil, Density, Cut


def boiling_point(max_cuts, api):
    '''
    return an array of boiling points for each psuedo-components
    Assume max_cuts * 2 components containing [saturate, aromatic]
    Output boiling point in this form:

      components: [s_0, a_0, s_1, a_1, ..., s_n, a_n]
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
    T_o = 457.16 - 3.3447 * api
    dT_dF = 1356.7 - 247.36 * log(api)

    bp = [float('nan')] * (max_cuts * 2)
    array_size = (max_cuts * 2)
    for ix in range(0, max_cuts * 2, 2):
        bp[ix] = dT_dF*(ix + 1)/array_size + T_o
        bp[ix + 1] = bp[ix]

    return bp


def sample_oil_to_mock_oil(max_cuts=None, **kwargs):
    '''
    make an Oil object from _sample_oils
    Currently, this has only been tested on sample oils, but should be made
    more general. Assume the kwargs are attributes of Oil object

    This adds following attributes:
    'densities' list containing only one density
    'cuts' list containing equal mass in saturates/aromatics for max cuts
    'resins' and 'asphaltene_content' to None
    '''
    if max_cuts is None:
        max_cuts = 5

    oil = Oil(**kwargs)

    # need to add densities list
    oil.densities = [
        Density(kg_m_3=uc.convert('density', 'api', 'kg/m^3', oil.api),
                ref_temp_k=288.16)]

    # add cuts - all mass goes into saturates/aromatics for now
    oil.resins = None
    oil.asphaltene_content = None
    mass_left = 1.0

    if oil.resins:
        mass_left -= oil.resins

    if oil.asphaltene_content:
        mass_left -= oil.resins

    # using two for loops but that's ok since this is just for testing/etc
    # at present.
    num_comp = max_cuts * 2
    bp_array = boiling_point(max_cuts, oil.api)
    oil.cuts = []
    mass_per_comp = mass_left / num_comp
    prev_mass_frac = 0.0
    for ix in range(num_comp):
        oil.cuts.append(Cut(fraction=prev_mass_frac + mass_per_comp,
                            vapor_temp_k=bp_array[ix]))
        prev_mass_frac += mass_per_comp

    return oil
