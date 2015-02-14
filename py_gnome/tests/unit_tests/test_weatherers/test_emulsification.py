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
                              Evaporation)
from gnome.outputters import WeatheringOutput
from gnome.spill.elements import floating

from ..conftest import sample_sc_release, sample_model_weathering, sample_model_weathering2


water = Water()
wind = constant_wind(15., 0)	#also test with lower wind no emulsification
waves = Waves(wind,water)

arrays = Emulsification().array_types


@pytest.mark.parametrize(('oil', 'temp', 'num_elems', 'on'),
                         [('ALBERTA', 311.15, 3, True),
                          ('BREGA', 311.15, 3, True),
                          ('FUEL OIL NO.6', 311.15, 3, False)])
def test_emulsification(oil, temp, num_elems, on):
    '''
    Fuel Oil #6 does not emulsify
    '''
    et = floating(substance=oil)
    sc = sample_sc_release(num_elements=num_elems,
                           element_type=et,
                           arr_types=arrays)
    time_step = 15. * 60
    model_time = (sc.spills[0].get('release_time') +
                  timedelta(seconds=time_step))

    emul = Emulsification(waves)
    emul.on = on

    emul.prepare_for_model_run(sc)

    # also want a test for a user set value for bulltime or bullwinkle
    if oil=='ALBERTA':
        sc['frac_lost'][:] = .31
    if oil=='BREGA':
        sc['frac_lost'][:] = .23
    #sc['frac_lost'][:] = .35
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

@pytest.mark.parametrize(('oil', 'temp'), [('ALBERTA', 333.0),
                                           ('FUEL OIL NO.6', 333.0),
                                           ('BREGA', 311.15),
                                           ])
def test_full_run(sample_model_fcn, oil, temp, dump):
    '''
    test evapoartion outputs post step for a full run of model. Dump json
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
                    np.allclose(sc.weathering_data['water_content'], 0.9))
            print ("Water fraction: {0}".
                   format(sc.weathering_data['water_content']))
            print "Completed step: {0}\n".format(step['step_num'])

    m_json_ = model.serialize('webapi')
    dump_json = os.path.join(dump, 'weathering_model.json')
    with open(dump_json, 'w') as f:
        json.dump(m_json_, f, indent=True)


def test_full_run_emul_not_active(sample_model_fcn):
    'no water/wind/waves object and no evaporation'
    model = sample_model_weathering(sample_model_fcn, 'oil_conservative')
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
