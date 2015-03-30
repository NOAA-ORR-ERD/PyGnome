'''
added this module for constructing an Oil object from a dict
It could be duck typed, though right now it defines a models.Oil object
Currently only works/tested for _sample_oils dict which only contains name
and API - needs work, but will suffice for testing
'''
import numpy
np = numpy

import unit_conversion as uc

from .models import Oil, KVis, Density, Cut
from .utilities import get_boiling_points_from_api

from .init_oil import (add_asphaltene_fractions,
                       add_resin_fractions,
                       add_molecular_weights,
                       add_saturate_aromatic_fractions,
                       add_component_densities,
                       adjust_resin_asphaltene_fractions)


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
    oil.densities = [Density(kg_m_3=uc.convert('density', 'api', 'kg/m^3',
                                               oil.api),
                             ref_temp_k=288.15)]

    if 'kvis' in kwargs:
        for k in kwargs['kvis']:
            oil.kvis.append(KVis(**k))

    add_resin_fractions(None, oil)
    add_asphaltene_fractions(None, oil)

    # add cuts - all mass goes into saturates/aromatics for now
    mass_left = 1.0
    mass_left -= sum([f.fraction for f in oil.sara_fractions
                      if f.sara_type in ('Resins', 'Asphaltenes')])

    oil.cuts = []
    prev_mass_frac = 0.0

    summed_boiling_points = []
    for t, f in get_boiling_points_from_api(max_cuts, mass_left, oil.api):
        added_to_sums = False

        for idx, [ut, summed_value] in enumerate(summed_boiling_points):
            if np.isclose(t, ut):
                summed_boiling_points[idx][1] += f
                added_to_sums = True
                break

        if added_to_sums is False:
            summed_boiling_points.append([t, f])

    for t_i, fraction in summed_boiling_points:
        oil.cuts.append(Cut(fraction=prev_mass_frac + fraction,
                            vapor_temp_k=t_i))
        prev_mass_frac += fraction

    add_molecular_weights(None, oil)
    add_saturate_aromatic_fractions(None, oil)
    add_component_densities(None, oil)
    adjust_resin_asphaltene_fractions(None, oil)

    return oil
