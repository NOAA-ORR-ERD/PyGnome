'''
Test evaporation module
'''
from datetime import timedelta

import pytest
import numpy as np

from gnome.environment import constant_wind, Water
from gnome.weatherers import Evaporation, Burn, Skimmer, Dispersion
from gnome.outputters import WeatheringOutput
from gnome.spill.elements import floating_weathering
from gnome.array_types import (mass_components,
                               windages,
                               density,
                               thickness,
                               mol,
                               evap_decay_constant)

from conftest import sample_sc_release, sample_model_weathering


water = Water()


arrays = {'windages': windages,
          'mass_components': mass_components,
          'density': density,
          'thickness': thickness,
          'mol': mol,
          'evap_decay_constant': evap_decay_constant}


@pytest.mark.parametrize(('oil', 'temp', 'num_elems', 'on'),
                         [('oil_conservative', 311.15, 3, True),
                          ('ALAMO', 311.15, 3, True),
                          ('FUEL OIL NO.6', 311.15, 3, False)])
def test_evaporation(oil, temp, num_elems, on):
    '''
    still working on tests ..
    '''
    et = floating_weathering(substance=oil)
    sc = sample_sc_release(num_elements=num_elems,
                           element_type=et,
                           arr_types=arrays)

    time_step = 15. * 60
    model_time = (sc.spills[0].get('release_time') +
                  timedelta(seconds=time_step))

    evap = Evaporation(water, wind=constant_wind(1., 0))
    if not on:
        evap.on = False

    evap.prepare_for_model_run(sc)
    evap.prepare_for_model_step(sc, time_step, model_time)
    init_mass = sc['mass_components'].copy()
    evap.weather_elements(sc, time_step, model_time)

    for spill in sc.spills:
        mask = sc.get_spill_mask(spill)
        bp = spill.get('substance').boiling_point
        if float('inf') in bp:
            sa = bp.index(float('inf'))
        else:
            sa = len(bp)
        if on:
            assert np.all(sc['evap_decay_constant'][mask, :sa] < 0.0)
            assert np.all(sc['evap_decay_constant'][mask, sa:] == 0.0)
        else:
            assert np.all(sc['evap_decay_constant'][mask, :] == 0.0)

        assert np.all(init_mass[mask, :sa] >=
                      sc['mass_components'][mask, :sa])
        assert np.all(init_mass[mask, sa:] ==
                      sc['mass_components'][mask, sa:])

    if on:
        assert sc.weathering_data['evaporated'] > 0.0
    else:
        assert sc.weathering_data['evaporated'] == 0.0
        assert np.all(sc['mass_components'] == init_mass)

    print sc['evap_decay_constant']
    print sc.weathering_data['evaporated']


def assert_helper(sc, new_p):
    'common assertions for spills and data in SpillContainer'
    for spill in sc.spills:
        mask = sc.get_spill_mask(spill)
        bp = spill.get('substance').boiling_point
        if float('inf') in bp:
            sa = bp.index(float('inf'))
        else:
            sa = len(bp)
        assert np.all(sc['evap_decay_constant'][mask, :sa] <= 0.0)
        assert np.all(sc['evap_decay_constant'][mask, sa:] == 0.0)
        assert np.allclose(np.sum(sc['mass_components'][mask, :], 1),
                           sc['mass'][mask])
        # not an instantaneous release so following is true even at step 0
        assert np.all(sc['mass'][mask] < spill.get_mass('kg'))

    if len(sc) > new_p:
        assert np.all(sc['evap_decay_constant'][:(len(sc)-new_p), :sa] < 0.0)
    if new_p > 0:
        assert np.all(sc['evap_decay_constant'][-new_p:, :sa] == 0.0)


@pytest.mark.parametrize(('oil', 'temp'), [('oil_conservative', 333.0),
                                           ('FUEL OIL NO.6', 333.0),
                                           ('ALAMO', 311.15),
                                           ])
def test_full_run(sample_model_fcn, oil, temp):
    model = sample_model_weathering(sample_model_fcn, oil, temp)
    model.environment += [Water(temp), constant_wind(1., 0)]
    model.weatherers += [Evaporation(model.environment[0],
                                     model.environment[1]),
                         Dispersion(),
                         Burn(),
                         Skimmer()]
    released = 0
    for step in model:
        for sc in model.spills.items():
            assert_helper(sc, sc.num_released - released)
            released = sc.num_released
            assert sc.weathering_data['floating'] == np.sum(sc['mass'])
            print ("Amount released: {0}".
                   format(sc.weathering_data['amount_released']))
            print "Mass floating: {0}".format(sc.weathering_data['floating'])
            print "Completed step: {0}\n".format(step['step_num'])


def test_full_run_evap_not_active(sample_model_fcn):
    'no water/wind object'
    model = sample_model_weathering(sample_model_fcn, 'oil_conservative')
    model.weatherers += Evaporation(on=False)
    model.outputters += WeatheringOutput()
    for step in model:
        for key in ('nominal', 'low', 'high'):
            assert 'floating' in step['WeatheringOutput'][key]
            assert 'amount_released' in step['WeatheringOutput'][key]
            assert 'evaporated' not in step['WeatheringOutput'][key]

        print ("Completed step: {0}"
               .format(step['WeatheringOutput']['step_num']))
