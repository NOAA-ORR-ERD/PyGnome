'''
Test evaporation module
'''
from datetime import timedelta

import pytest
import numpy as np

from gnome.environment import constant_wind, Water
from gnome.weatherers import Evaporation
from gnome.spill.elements import floating_weathering
from gnome.spill import point_line_release_spill
from gnome.array_types import (mass_components,
                               windages,
                               density,
                               thickness,
                               mol,
                               evap_decay_constant)

from conftest import sample_sc_release


water = Water()


def assert_helper(sc):
    'common assertions for spills and data in SpillContainer'
    for spill in sc.spills:
        mask = sc.get_spill_mask(spill)
        bp = spill.get('substance').boiling_point
        if float('inf') in bp:
            sa = bp.index(float('inf'))
        else:
            sa = len(bp)
        assert np.all(sc['evap_decay_constant'][mask, :sa] < 0.0)
        assert np.all(sc['evap_decay_constant'][mask, sa:] == 0.0)
        assert np.all(np.sum(sc['mass_components'][mask, :], 1) ==
                      sc['mass'][mask])
        assert np.all(sc['mass'][mask] < spill.get_mass('kg'))


@pytest.mark.parametrize(('oil', 'temp', 'num_elems'),
                         [('oil_conservative', 311.15, 3),
                          ('FUEL OIL NO.6', 311.15, 3)])
def test_evaporation(oil, temp, num_elems):
    '''
    still working on tests ..
    '''
    et = floating_weathering(substance=oil)
    arrays = {'windages': windages,
              'mass_components': mass_components,
              'density': density,
              'thickness': thickness,
              'mol': mol,
              'evap_decay_constant': evap_decay_constant}

    sc = sample_sc_release(num_elements=num_elems,
                           element_type=et,
                           arr_types=arrays)

    time_step = 15. * 60
    model_time = (sc.spills[0].get('release_time') +
                  timedelta(seconds=time_step))

    evap = Evaporation(water, wind=constant_wind(1., 0))
    evap.prepare_for_model_run(sc)
    evap.prepare_for_model_step(sc, time_step, model_time)
    mass_remain = evap.weather_elements(sc, time_step, model_time)

    for spill in sc.spills:
        mask = sc.get_spill_mask(spill)
        bp = spill.get('substance').boiling_point
        if float('inf') in bp:
            sa = bp.index(float('inf'))
        else:
            sa = len(bp)
        assert np.all(sc['evap_decay_constant'][mask, :sa] < 0.0)
        assert np.all(sc['evap_decay_constant'][mask, sa:] == 0.0)
        assert np.all(mass_remain[mask, :sa] <=
                      sc['mass_components'][mask, :sa])
        assert np.all(mass_remain[mask, sa:] ==
                      sc['mass_components'][mask, sa:])

    print sc['evap_decay_constant']
    print mass_remain


@pytest.mark.parametrize(('oil', 'temp'), [('oil_conservative', 333.0),
                                           ('FUEL OIL NO.6', 333.0)
                                           ])
def test_full_run(sample_model_fcn, oil, temp):
    model = sample_model_fcn['model']
    model.uncertain = False     # fixme: with uncertainty, copying spill fails!
    et = floating_weathering(substance=oil)
    spill = point_line_release_spill(1,
                                     sample_model_fcn['release_start_pos'],
                                     model.start_time,
                                     element_type=et,
                                     amount=100,
                                     units='kg')
    model.spills += spill
    model.environment += [Water(temp), constant_wind(1., 0)]
    model.weatherers += Evaporation(model.environment[0], model.environment[1])
    for step in model:
        if step['step_num'] == 0:
            continue

        print "Completed step: {0}".format(step['step_num'])
        for sc in model.spills.items():
            assert_helper(sc)
            print "Mass remaining: {0}".format(np.sum(sc['mass'], 0))
