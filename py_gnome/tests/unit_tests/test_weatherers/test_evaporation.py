'''
Test evaporation module
'''
import os
import json
from datetime import timedelta

import pytest
import numpy as np

from gnome.environment import constant_wind, Water, Wind
from gnome.weatherers import (Evaporation,
                              Burn,
                              Skimmer,
                              Dispersion,
                              IntrinsicProps)
from gnome.outputters import WeatheringOutput
from gnome.spill.elements import floating_weathering
from gnome.array_types import (mass_components,
                               windages,
                               thickness,
                               mol,
                               evap_decay_constant)
from gnome.basic_types import oil_status

from ..conftest import sample_sc_release, sample_model_weathering
from . import dump

dump_json = os.path.join(dump, 'weathering_model.json')
water = Water()

arrays = Evaporation().array_types
intrinsic = IntrinsicProps(water, arrays)
arrays.update(intrinsic.array_types)


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
    intrinsic.update(sc.num_released, sc)
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
            mask = sc['status_codes'] == oil_status.in_water
            assert sc.weathering_data['floating'] == np.sum(sc['mass'][mask])
            print ("Amount released: {0}".
                   format(sc.weathering_data['amount_released']))
            print "Mass floating: {0}".format(sc.weathering_data['floating'])
            print "Completed step: {0}\n".format(step['step_num'])

    m_json_ = model.serialize('webapi')
    with open(dump_json, 'w') as f:
        json.dump(m_json_, f, indent=True)


def test_full_run_evap_not_active(sample_model_fcn):
    'no water/wind object'
    model = sample_model_weathering(sample_model_fcn, 'oil_conservative')
    model.weatherers += Evaporation(on=False)
    model.outputters += WeatheringOutput()
    for step in model:
        for key in step['WeatheringOutput']:
            if key not in ('step_num', 'time_stamp'):
                assert 'floating' in step['WeatheringOutput'][key]
                assert 'amount_released' in step['WeatheringOutput'][key]
                assert 'evaporated' not in step['WeatheringOutput'][key]

        print ("Completed step: {0}"
               .format(step['WeatheringOutput']['step_num']))


def test_serialize_deseriailize():
    'test serialize/deserialize for webapi'
    e = Evaporation()
    wind = constant_wind(1., 0)
    water = Water()
    json_ = e.serialize()
    json_['wind'] = wind.serialize()
    json_['water'] = water.serialize()

    # deserialize and ensure the dict's are correct
    d_ = Evaporation.deserialize(json_)
    assert d_['wind'] == Wind.deserialize(json_['wind'])
    assert d_['water'] == Water.deserialize(json_['water'])
    d_['wind'] = wind
    d_['water'] = water
    e.update_from_dict(d_)
    assert e.wind is wind
    assert e.water is water
