'''
Test emulsification module
'''
import os
import json
from datetime import timedelta

import pytest
import numpy as np

from gnome.environment import constant_wind, Water, Waves
from gnome.weatherers import (Emulsification,
                              Evaporation,
                              WeatheringData)
from gnome.outputters import WeatheringOutput
from gnome.spill.elements import floating

from ..conftest import (sample_sc_release,
                        sample_model_weathering,
                        sample_model_weathering2,
                        test_oil)


water = Water()
wind = constant_wind(15., 0)	#also test with lower wind no emulsification
waves = Waves(wind, water)

arrays = Emulsification().array_types
intrinsic = WeatheringData(water)
arrays.update(intrinsic.array_types)

# need an oil that emulsifies and one that does not
#s_oils = [test_oil, 'FUEL OIL NO.6']
s_oils = [test_oil, test_oil]


@pytest.mark.parametrize(('oil', 'temp', 'num_elems', 'on'),
                         [(s_oils[0], 311.15, 3, True),
                          (s_oils[1], 311.15, 3, False)])
def test_emulsification(oil, temp, num_elems, on):
    '''
    Fuel Oil #6 does not emulsify
    '''
    et = floating(substance=oil)
    time_step = 15. * 60
    sc = sample_sc_release(num_elements=num_elems,
                           element_type=et,
                           arr_types=arrays,
                           time_step=time_step)
    intrinsic.update(sc.num_released, sc, time_step)
    model_time = (sc.spills[0].get('release_time') +
                  timedelta(seconds=time_step))

    emul = Emulsification(waves)
    emul.on = on

    emul.prepare_for_model_run(sc)

    # also want a test for a user set value for bulltime or bullwinkle
    if oil == s_oils[0]:
        sc['frac_lost'][:] = .31

    # sc['frac_lost'][:] = .35
    print "sc['frac_lost'][:]"
    print sc['frac_lost'][:]
    emul.prepare_for_model_step(sc, time_step, model_time)
    emul.weather_elements(sc, time_step, model_time)
    print "sc['frac_water'][:]"
    print sc['frac_water'][:]

    if on:
        assert np.all(sc['frac_lost'] > 0) and np.all(sc['frac_lost'] < 1.0)
        assert np.all(sc['frac_water'] > 0) and np.all(sc['frac_water'] <= .9)
    else:
        assert np.all(sc['frac_water'] == 0)

    sc['frac_lost'][:] = .2
    print "sc['frac_lost'][:]"
    print sc['frac_lost'][:]
    emul.prepare_for_model_step(sc, time_step, model_time)
    emul.weather_elements(sc, time_step, model_time)
    print "sc['frac_water'][:]"
    print sc['frac_water'][:]

    assert np.all(sc['frac_water'] == 0)


@pytest.mark.parametrize(('oil', 'temp'), [(s_oils[0], 333.0),
                                           (s_oils[1], 333.0),
                                           ])
def test_full_run(sample_model_fcn, oil, temp):
    '''
    test emulsification outputs post step for a full run of model. Dump json
    for 'weathering_model.json' in dump directory
    '''
    model = sample_model_weathering2(sample_model_fcn, oil, temp)
    model.water = Water(temp)
    model.environment += [waves, constant_wind(15., 0)]
    model.weatherers += Evaporation(model.water, model.environment[1])
    model.weatherers += Emulsification(model.environment[0])

    for step in model:
        for sc in model.spills.items():
            # need or condition to account for water_content = 0.9000000000012
            # or just a little bit over 0.9
            assert (sc.weathering_data['water_content'] <= .9 or
                    np.isclose(sc.weathering_data['water_content'], 0.9))
            print ("Water fraction: {0}".
                   format(sc.weathering_data['water_content']))
            print "Completed step: {0}\n".format(step['step_num'])


def test_full_run_emul_not_active(sample_model_fcn):
    'no water/wind/waves object and no evaporation'
    model = sample_model_weathering(sample_model_fcn, 'oil_crude')
    model.weatherers += Emulsification(on=False)
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


def test_bulltime():
    '''
    user set time to start emulsification
    '''

    et = floating(substance=test_oil)
    assert et.substance.bulltime == -999
    et.substance.bulltime = 3600
    assert et.substance.bulltime == 3600


def test_bullwinkle():
    '''
    user set emulsion constant
    '''

    et = floating(substance=test_oil)
    assert et.substance.bullwinkle == .303
    et.substance.bullwinkle = .4
    assert et.substance.bullwinkle == .4

def test_serialize_deseriailize():
    'test serialize/deserialize for webapi'
    wind = constant_wind(15., 0)
    waves = Waves(wind, Water())
    e = Emulsification(waves)
    json_ = e.serialize()
    json_['waves'] = waves.serialize()

    # deserialize and ensure the dict's are correct
    d_ = Emulsification.deserialize(json_)
    assert d_['waves'] == Waves.deserialize(json_['waves'])
    d_['waves'] = waves
    e.update_from_dict(d_)
    assert e.waves is waves
