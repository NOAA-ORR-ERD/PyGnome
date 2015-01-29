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
from gnome.spill.elements import floating
from gnome.array_types import (mass_components,
                               windages,
                               thickness,
                               mol,
                               evap_decay_constant)
from gnome.basic_types import oil_status

from ..conftest import sample_sc_release, sample_model_weathering


water = Water()

arrays = Evaporation().array_types
intrinsic = IntrinsicProps(water)
arrays.update(intrinsic.array_types)


@pytest.mark.parametrize(('oil', 'temp', 'num_elems', 'on'),
                         [('AGUA DULCE', 311.15, 3, True),
                          ('ALAMO', 311.15, 3, True),
                          ('FUEL OIL NO.6', 311.15, 3, False)])
def test_evaporation(oil, temp, num_elems, on):
    '''
    still working on tests ..
    '''
    et = floating(substance=oil)
    sc = sample_sc_release(num_elements=num_elems,
                           element_type=et,
                           arr_types=arrays)
    intrinsic.update(sc.num_released, sc)
    time_step = 15. * 60
    model_time = (sc.spills[0].get('release_time') +
                  timedelta(seconds=time_step))

    evap = Evaporation(water, wind=constant_wind(1., 0))
    evap.on = on

    evap.prepare_for_model_run(sc)
    evap.prepare_for_model_step(sc, time_step, model_time)
    init_mass = sc['mass_components'].copy()
    evap.weather_elements(sc, time_step, model_time)

    if on:
        assert np.all(sc['frac_lost'] > 0) and np.all(sc['frac_lost'] < 1.0)

        # all elements experience the same evaporation
        assert np.all(sc['frac_lost'][0] == sc['frac_lost'])

    for spill in sc.spills:
        mask = sc.get_spill_mask(spill)
        sa = -2     # last two elements are now always resins and asphaltenes 
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

    print '\nevap_decay_const', sc['evap_decay_constant']
    print 'frac_lost', sc['frac_lost']
    print 'total evaporated', sc.weathering_data['evaporated']


def assert_helper(sc, new_p):
    'common assertions for spills and data in SpillContainer'
    total_mass = sum([spill.get_mass('kg') for spill in sc.spills])
    arrays = ['evap_decay_constant', 'mass_components', 'mass', 'status_codes']
    for substance, data in sc.itersubstancedata(arrays):
        # resins and asphaltenes are always present in data now
        sa = -2

        if len(sc) > new_p:
            old_le = len(sc)-new_p
            inwater = data['status_codes'][:old_le] == oil_status.in_water
            assert np.all(data['evap_decay_constant'][:old_le, :sa][inwater] <
                          0.0)
            assert np.all(data['evap_decay_constant'][:old_le, :sa][~inwater]
                          == 0.0)
            # heavy components always have evap_decay_constant of 0.0
            assert np.all(data['evap_decay_constant'][:old_le, sa:] == 0.0)
            assert np.allclose(np.sum(data['mass_components'], 1),
                               data['mass'])
            # not an instantaneous release so following is true even at step 0
            assert data['mass'].sum() < total_mass

        if new_p > 0:
            assert np.all(data['evap_decay_constant'][-new_p:, :] ==
                          0.0)


@pytest.mark.parametrize(('oil', 'temp'), [('AGUA DULCE', 333.0),
                                           ('FUEL OIL NO.6', 333.0),
                                           ('ALAMO', 311.15),
                                           ])
def test_full_run(sample_model_fcn, oil, temp, dump):
    '''
    test evapoartion outputs post step for a full run of model. Dump json
    for 'weathering_model.json' in dump directory
    This contains a mover so at some point several elements end up on_land.
    This test also checks the evap_decay_constant for elements that are not
    in water is 0 so mass is unchanged.
    '''
    model = sample_model_weathering(sample_model_fcn, oil, temp)
    model.environment += [Water(temp), constant_wind(1., 0)]
    model.weatherers += [Evaporation(model.environment[0],
                                     model.environment[1])]
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
            print "Mass evap: {0}".format(sc.weathering_data['evaporated'])
            print "LEs in water: {0}".format(sum(mask))
            print "Mass on land: {0}".format(np.sum(sc['mass'][~mask]))

            print "Completed step: {0}\n".format(step['step_num'])

    m_json_ = model.serialize('webapi')
    dump_json = os.path.join(dump, 'weathering_model.json')
    with open(dump_json, 'w') as f:
        json.dump(m_json_, f, indent=True)


def test_full_run_evap_not_active(sample_model_fcn):
    'no water/wind object'
    model = sample_model_weathering(sample_model_fcn, 'oil_conservative')
    model.weatherers += Evaporation(on=False)
    model.outputters += WeatheringOutput()
    for step in model:
        '''
        if no weatherers, then no weathering output - need to add on/off
        switch to WeatheringOutput
        '''
        assert len(step['WeatheringOutput']) == 2
        assert ('step_num' in step['WeatheringOutput'] and
                'time_stamp' in step['WeatheringOutput'])
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
