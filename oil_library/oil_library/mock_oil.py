'''
added this module for constructing an Oil object from a dict
It could be duck typed, though right now it defines a models.Oil object
Currently only works/tested for _sample_oils dict which only contains name
and API - needs work, but will suffice for testing
'''
from hazpy import unit_conversion
uc = unit_conversion

from .models import Oil, Density, Cut
from .utilities import get_boiling_points_from_api


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

    oil.cuts = []
    prev_mass_frac = 0.0
    for t_i, fraction in get_boiling_points_from_api(max_cuts, mass_left,
                                                     oil.api):
        oil.cuts.append(Cut(fraction=prev_mass_frac + fraction,
                            vapor_temp_k=t_i))
        prev_mass_frac += fraction

    return oil
