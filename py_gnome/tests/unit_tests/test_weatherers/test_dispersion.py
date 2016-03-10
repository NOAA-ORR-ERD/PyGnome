'''
Test natural dispersion module
'''
from datetime import timedelta

import pytest
import numpy as np

from gnome.environment import constant_wind, Water, Waves
from gnome.weatherers import (NaturalDispersion,
                              Evaporation,
                              Emulsification)
from gnome.outputters import WeatheringOutput
from gnome.spill.elements import floating

from conftest import weathering_data_arrays
from ..conftest import (sample_model_weathering,
                        sample_model_weathering2)


water = Water()
# also test with lower wind no dispersion
wind = constant_wind(15., 270, 'knots')
waves = Waves(wind, water)


@pytest.mark.parametrize(('oil', 'temp', 'num_elems', 'on'),
                         [('ABU SAFAH', 311.15, 3, True),
                          ('BAHIA', 311.15, 3, True),
                          ('ALASKA NORTH SLOPE (MIDDLE PIPELINE)', 311.15, 3,
                           False)])
def test_dispersion(oil, temp, num_elems, on):
    '''
    Fuel Oil #6 does not exist...
    '''
    et = floating(substance=oil)
    disp = NaturalDispersion(waves, water)
    (sc, time_step) = weathering_data_arrays(disp.array_types,
                                             water,
                                             element_type=et)[:2]
    model_time = (sc.spills[0].get('release_time') +
                  timedelta(seconds=time_step))

    disp.on = on
    disp.prepare_for_model_run(sc)

    disp.prepare_for_model_step(sc, time_step, model_time)
    disp.weather_elements(sc, time_step, model_time)

    if on:
        assert sc.mass_balance['natural_dispersion'] > 0
        assert sc.mass_balance['sedimentation'] > 0
        print "sc.mass_balance['natural_dispersion']"
        print sc.mass_balance['natural_dispersion']
        print "sc.mass_balance['sedimentation']"
        print sc.mass_balance['sedimentation']
    else:
        assert 'natural_dispersion' not in sc.mass_balance
        assert 'sedimentation' not in sc.mass_balance


@pytest.mark.parametrize(('oil', 'temp', 'num_elems'),
                         [('ABU SAFAH', 288.15, 3)])
def test_dispersion_not_active(oil, temp, num_elems):
    '''
    Fuel Oil #6 does not exist...
    '''
    disp = NaturalDispersion(waves, water)
    (sc, time_step) = \
        weathering_data_arrays(disp.array_types,
                               water,
                               element_type=floating(substance=oil))[:2]
    sc.amount = 10000
    model_time = (sc.spills[0].get('release_time') +
                  timedelta(seconds=time_step))

    disp.prepare_for_model_run(sc)

    new_model_time = (sc.spills[0].get('release_time') +
                      timedelta(seconds=3600))

    disp.active_start = new_model_time
    disp.prepare_for_model_step(sc, time_step, model_time)
    disp.weather_elements(sc, time_step, model_time)

    assert np.all(sc.mass_balance['natural_dispersion'] == 0)
    assert np.all(sc.mass_balance['sedimentation'] == 0)


@pytest.mark.parametrize(('oil', 'temp', 'dispersed'),
                         [('ABU SAFAH', 288.7, 34.909),
                          ('ALASKA NORTH SLOPE (MIDDLE PIPELINE)',
                           288.7, 162.454),
                          ('BAHIA', 288.7, 89.400)
                          ]
                         )
def test_full_run(sample_model_fcn2, oil, temp, dispersed):
    '''
    test dispersion outputs post step for a full run of model. Dump json
    for 'weathering_model.json' in dump directory
    '''
    model = sample_model_weathering2(sample_model_fcn2, oil, temp)
    model.environment += [Water(temp), wind,  waves]
    model.weatherers += Evaporation()
    model.weatherers += Emulsification(waves)
    model.weatherers += NaturalDispersion()

    # set make_default_refs to True for objects contained in model after adding
    # objects to the model
    model.set_make_default_refs(True)

    for step in model:
        for sc in model.spills.items():
            if step['step_num'] > 0:
                assert (sc.mass_balance['natural_dispersion'] > 0)
                assert (sc.mass_balance['sedimentation'] > 0)
            print ("Dispersed: {0}".
                   format(sc.mass_balance['natural_dispersion']))
            print ("Sedimentation: {0}".
                   format(sc.mass_balance['sedimentation']))
            print "Completed step: {0}\n".format(step['step_num'])

    sc = model.spills.items()[0]
    print (sc.mass_balance['natural_dispersion'], dispersed)
    assert np.isclose(sc.mass_balance['natural_dispersion'], dispersed,
                      atol=0.001)


def test_full_run_disp_not_active(sample_model_fcn):
    'no water/wind/waves object and no evaporation'
    model = sample_model_weathering(sample_model_fcn, 'oil_6')
    model.weatherers += NaturalDispersion(on=False)
    model.outputters += WeatheringOutput()
    for step in model:
        '''
        if no weatherers, then no weathering output - need to add on/off
        switch to WeatheringOutput
        '''
        assert 'natural_dispersion' not in step['WeatheringOutput']
        assert 'sedimentation' not in step['WeatheringOutput']
        assert ('time_stamp' in step['WeatheringOutput'])
        print ("Completed step: {0}"
               .format(step['step_num']))


def test_serialize_deseriailize():
    'test serialize/deserialize for webapi'
    wind = constant_wind(15., 0)
    waves = Waves(wind, Water())
    e = NaturalDispersion(waves)
    json_ = e.serialize()
    json_['waves'] = waves.serialize()

    # deserialize and ensure the dict's are correct
    d_ = NaturalDispersion.deserialize(json_)
    assert d_['waves'] == Waves.deserialize(json_['waves'])
    d_['waves'] = waves
    e.update_from_dict(d_)
    assert e.waves is waves
