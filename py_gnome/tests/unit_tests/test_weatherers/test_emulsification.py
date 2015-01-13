'''
Test emulsification module
'''
import os
import json
from datetime import timedelta

import pytest
import numpy as np

from gnome.environment import constant_wind, Water, Waves, Wind
from gnome.weatherers import (Emulsification,
                              Evaporation,
                              Burn,
                              Skimmer,
                              Dispersion)
from gnome.outputters import WeatheringOutput
from gnome.spill.elements import floating

from ..conftest import sample_sc_release, sample_model_weathering, sample_model_weathering2


water = Water()
wind=constant_wind(15., 0)	#also test with lower wind no emulsification
waves = Waves(wind,water)

arrays = Emulsification().array_types


@pytest.mark.parametrize(('oil', 'temp', 'num_elems', 'on'),
                         [('AGUA DULCE', 311.15, 3, True),
                          ('ALAMO', 311.15, 3, True),
                          ('FUEL OIL NO.6', 311.15, 3, False)])
def test_emulsification(oil, temp, num_elems, on):
    '''
    still working on tests ..
    '''
    et = floating(substance=oil)
    sc = sample_sc_release(num_elements=num_elems,
                           element_type=et,
                           arr_types=arrays)
    time_step = 15. * 60
    model_time = (sc.spills[0].get('release_time') +
                  timedelta(seconds=time_step))

    emul = Emulsification(waves, wind)
    emul.on = on

    emul.prepare_for_model_run(sc)

    # also want a test for a user set value for bulltime or bullwinkle
    if oil=='ALAMO':
        sc['frac_lost'][:] = .8
    if oil=='AGUA DULCE':
        sc['frac_lost'][:] = .92
    #sc['frac_lost'][:] = .92
    #print "sc['frac_lost'][:]"
    #print sc['frac_lost'][:]
    emul.prepare_for_model_step(sc, time_step, model_time)
    emul.weather_elements(sc, time_step, model_time)
    #print "sc['frac_water'][:]"
    #print sc['frac_water'][:]

    if on:
        assert np.all(sc['frac_lost'] > 0) and np.all(sc['frac_lost'] < 1.0)
        assert np.all(sc['frac_water'] > 0) and np.all(sc['frac_water'] <= .9)
    else:
        #assert np.all(sc['frac_lost'] > 0) and np.all(sc['frac_lost'] < 1.0)
        assert np.all(sc['frac_water'] == 0)

    sc['frac_lost'][:] = .75
    #print "sc['frac_lost'][:]"
    #print sc['frac_lost'][:]
    emul.prepare_for_model_step(sc, time_step, model_time)
    emul.weather_elements(sc, time_step, model_time)
    #print "sc['frac_water'][:]"
    #print sc['frac_water'][:]

    assert np.all(sc['frac_water'] == 0)

@pytest.mark.parametrize(('oil', 'temp'), [('AGUA DULCE', 333.0),
                                           ('FUEL OIL NO.6', 333.0),
                                           ('ALAMO', 311.15),
                                           ])
def test_full_run(sample_model_fcn, oil, temp, dump):
    '''
    test evapoartion outputs post step for a full run of model. Dump json
    for 'weathering_model.json' in dump directory
    '''
    model = sample_model_weathering2(sample_model_fcn, oil, temp)
    model.environment += [Water(temp), waves, constant_wind(15., 0)]
    model.weatherers += [Emulsification(model.environment[1],
                                     model.environment[2]),
                         Evaporation(model.environment[0],
                                     model.environment[2]),
                         Dispersion(),
                         Burn(),
                         Skimmer()]
    released = 0
    for step in model:
        for sc in model.spills.items():
            #assert_helper(sc, sc.num_released - released)
            #released = sc.num_released
            #mask = sc['status_codes'] == oil_status.in_water
            assert sc.weathering_data['emulsified'] < 1
            print ("Water fraction: {0}".
                   format(sc.weathering_data['emulsified']))
            #print "Mass floating: {0}".format(sc.weathering_data['floating'])
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
    e = Emulsification(waves,wind)
    json_ = e.serialize()
    json_['wind'] = wind.serialize()
    json_['waves'] = waves.serialize()

    # deserialize and ensure the dict's are correct
    d_ = Emulsification.deserialize(json_)
    assert d_['wind'] == Wind.deserialize(json_['wind'])
    assert d_['waves'] == Waves.deserialize(json_['waves']) 
    d_['wind'] = wind
    d_['waves'] = waves
    e.update_from_dict(d_)
    assert e.wind is wind
    assert e.waves is waves
