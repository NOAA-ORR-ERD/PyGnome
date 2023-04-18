'''
Test natural dispersion module
'''
from datetime import timedelta

import pytest
import numpy as np

from gnome.utilities.inf_datetime import InfDateTime

from gnome.environment import constant_wind, Water, Waves
from gnome.weatherers import (NaturalDispersion,
                              Evaporation,
                              Emulsification)
from gnome.outputters import WeatheringOutput

from .conftest import weathering_data_arrays
from ..conftest import (sample_model_weathering,
                        sample_model_weathering2)


water = Water()
# also test with lower wind no dispersion
wind = constant_wind(15., 270, 'knots')
waves = Waves(wind, water)


@pytest.mark.parametrize(('oil', 'temp', 'num_elems', 'on'),
                         [('oil_bahia', 311.15, 3, True),
                          # ('BAHIA', 311.15, 3, True),
                          # ('ABU SAFAH', 311.15, 3, True),
                          ('oil_ans_mp', 311.15, 3, True),
                          # ('ALASKA NORTH SLOPE (MIDDLE PIPELINE)', 311.15, 3,
                          ('oil_ans_mp', 311.15, 3,
                           False)])
def test_dispersion(oil, temp, num_elems, on):
    '''
    Fuel Oil #6 does not exist...
    '''
    disp = NaturalDispersion(waves, water)
    (sc, time_step) = weathering_data_arrays(disp.array_types,
                                             water)[:2]
    model_time = (sc.spills[0].release_time +
                  timedelta(seconds=time_step))

    disp.on = on
    disp.prepare_for_model_run(sc)

    disp.prepare_for_model_step(sc, time_step, model_time)
    disp.weather_elements(sc, time_step, model_time)

    if on:
       # print("sc.mass_balance['natural_dispersion']")
       # print(sc.mass_balance['natural_dispersion'])
       # print("sc.mass_balance['sedimentation']")
       # print(sc.mass_balance['sedimentation'])

        assert sc.mass_balance['natural_dispersion'] > 0
        assert sc.mass_balance['sedimentation'] > 0
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
                               water)[:2]

    sc.amount = 10000
    model_time = (sc.spills[0].release_time +
                  timedelta(seconds=time_step))

    disp.prepare_for_model_run(sc)

    assert np.all(sc.mass_balance['natural_dispersion'] == 0)
    assert np.all(sc.mass_balance['sedimentation'] == 0)

    new_model_time = (sc.spills[0].release_time +
                      timedelta(seconds=3600))

    disp.active_range = (new_model_time, InfDateTime('inf'))
    disp.prepare_for_model_step(sc, time_step, model_time)

    assert np.all(sc.mass_balance['natural_dispersion'] == 0)
    assert np.all(sc.mass_balance['sedimentation'] == 0)

    disp.weather_elements(sc, time_step, model_time)

    assert np.all(sc.mass_balance['natural_dispersion'] == 0)
    assert np.all(sc.mass_balance['sedimentation'] == 0)


#@pytest.mark.xfail
# the test oils don't match the data base, using so tests don't depend on db
# Note: test values appear to be reasonable as Feb-13-2023
# Note: this test may change if any other weathering algorithms change
@pytest.mark.parametrize(('oil', 'temp', 'dispersed'),
#                          [('oil_bahia', 288.7, 264.076),
#                           ('oil_ans_mp', 288.7, 404.244),
#                           ]
                         [('oil_bahia', 288.7, 400.753),
                          ('oil_ans_mp', 288.7, 697.214),
                          ]
                         )
def test_full_run_DS1988(sample_model_fcn2, oil, temp, dispersed):
    '''
    test dispersion outputs post step for a full run of model. Dump json
    for 'weathering_model.json' in dump directory
    '''
    model = sample_model_weathering2(sample_model_fcn2, oil, temp)
    model.environment += [Water(temp), wind, waves]
    model.weatherers += Evaporation()
    model.weatherers += Emulsification(waves)
    model.weatherers += NaturalDispersion()

    # set make_default_refs to True for objects contained in model after adding
    # objects to the model
    model.set_make_default_refs(True)

    for step in model:
        for sc in list(model.spills.items()):
            if step['step_num'] > 0:
                # print ("Dispersed: {0}".
                #        format(sc.mass_balance['natural_dispersion']))
                # print ("Sedimentation: {0}".
                #        format(sc.mass_balance['sedimentation']))
                # print "Completed step: {0}\n".format(step['step_num'])

                assert (sc.mass_balance['natural_dispersion'] > 0)
                assert (sc.mass_balance['sedimentation'] > 0)

    sc = list(model.spills.items())[0]

    assert np.isclose(sc.mass_balance['natural_dispersion'], dispersed,
                      atol=0.001)

@pytest.mark.parametrize(('oil', 'temp', 'dispersed'),
#                          [('oil_bahia', 288.7, 4131.36),
#                           ('oil_ans_mp', 288.7, 8158.10),
#                           ]
                         [('oil_bahia', 288.7, 3399.69),
                          ('oil_ans_mp', 288.7, 7875.77),
                          ]
                         )
def test_full_run_Li2017(sample_model_fcn2, oil, temp, dispersed):
    '''
    test dispersion outputs post step for a full run of model. Dump json
    for 'weathering_model.json' in dump directory
    '''
    model = sample_model_weathering2(sample_model_fcn2, oil, temp)
    model.environment += [Water(temp), wind, waves]
    model.weatherers += Evaporation()
    model.weatherers += Emulsification(waves)
    model.weatherers += NaturalDispersion(algorithm='Li2017')

    # set make_default_refs to True for objects contained in model after adding
    # objects to the model
    model.set_make_default_refs(True)

    for step in model:
        for sc in list(model.spills.items()):
            if step['step_num'] > 0:
                # print ("Dispersed: {0}".
                #        format(sc.mass_balance['natural_dispersion']))
                # print ("Sedimentation: {0}".
                #        format(sc.mass_balance['sedimentation']))
                # print "Completed step: {0}\n".format(step['step_num'])

                assert (sc.mass_balance['natural_dispersion'] > 0)
                assert (sc.mass_balance['sedimentation'] > 0)

    sc = list(model.spills.items())[0]
    print('lllleeeeoooo', sc.mass_balance['natural_dispersion'])
    print('000000000000', dispersed)
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

        # print ("Completed step: {0}"
        #        .format(step['step_num']))

#@pytest.mark.skipif(reason="serialization for weatherers overall needs review")
def test_serialize_deseriailize():
    'test serialize/deserialize for webapi'
    wind = constant_wind(15., 0)
    waves = Waves(wind, Water())
    e = NaturalDispersion(waves, algorithm='Li2017')
    json_ = e.serialize()

    # deserialize and ensure the dict's are correct
    d_ = NaturalDispersion.deserialize(json_)
    assert d_.waves == Waves.deserialize(json_['waves'])
    assert d_.waves == waves
    assert d_.algorithm == 'Li2017'

