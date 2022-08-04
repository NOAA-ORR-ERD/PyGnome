'''
Test emulsification module
'''

from datetime import timedelta

import pytest
import numpy as np

from gnome.environment import constant_wind, Water, Waves
from gnome.weatherers import (Emulsification,
                              Evaporation)
from gnome.outputters import WeatheringOutput

from .conftest import weathering_data_arrays

from ..conftest import (sample_model_weathering,
                        sample_model_weathering2,
                        test_oil)
from gnome.spills.gnome_oil import GnomeOil


water = Water()
wind = constant_wind(15., 0)  # also test with lower wind no emulsification
waves = Waves(wind, water)

# need an oil that emulsifies and one that does not
# s_oils = [test_oil, 'FUEL OIL NO.6']
s_oils = [test_oil, test_oil]


@pytest.mark.parametrize(('oil', 'temp', 'num_elems', 'on'),
                         [(s_oils[0], 311.15, 3, True),
                          (s_oils[1], 311.15, 3, False)])
def test_emulsification(oil, temp, num_elems, on):
    '''
    Fuel Oil #6 does not emulsify
    fixme: this fails for ALASKA NORTH SLOPE - what is it supposed to test?
    '''
    print(oil, temp, num_elems, on)

    emul = Emulsification(waves)
    emul.on = on

    (sc, time_step) = \
        weathering_data_arrays(emul.array_types,
                               water)[:2]
    model_time = (sc.spills[0].release_time +
                  timedelta(seconds=time_step))

    emul.prepare_for_model_run(sc)

    # also want a test for a user set value for
    # bullwinkle_time or bullwinkle_fraction
    if oil == s_oils[0]:
        sc['frac_evap'][:] = .31

    # sc['frac_evap'][:] = .35
    print("sc['frac_evap'][:]")
    print(sc['frac_evap'][:])

    emul.prepare_for_model_step(sc, time_step, model_time)
    emul.weather_elements(sc, time_step, model_time)

    print("sc['frac_water'][:]")
    print(sc['frac_water'][:])

    if on:
        assert np.all(sc['frac_evap'] > 0) and np.all(sc['frac_evap'] < 1.0)
        assert np.all(sc['frac_water'] > 0) and np.all(sc['frac_water'] <= .9)
    else:
        assert np.all(sc['frac_water'] == 0)

    sc['frac_evap'][:] = .2
    print("sc['frac_evap'][:]")
    print(sc['frac_evap'][:])

    emul.prepare_for_model_step(sc, time_step, model_time)
    emul.weather_elements(sc, time_step, model_time)

    print("sc['frac_water'][:]")
    print(sc['frac_water'][:])

    if on:
        assert np.all(sc['frac_evap'] > 0) and np.all(sc['frac_evap'] < 1.0)
        assert np.all(sc['frac_water'] > 0) and np.all(sc['frac_water'] <= .9)
    else:
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
    model.environment += [Waves(), wind, Water(temp)]
    model.weatherers += Evaporation()
    model.weatherers += Emulsification()
    model.set_make_default_refs(True)

    for step in model:
        for sc in list(model.spills.items()):
            # need or condition to account for water_content = 0.9000000000012
            # or just a little bit over 0.9
            assert (sc.mass_balance['water_content'] <= .9 or
                    np.isclose(sc.mass_balance['water_content'], 0.9))
            print(("Water fraction: {0}".
                   format(sc.mass_balance['water_content'])))
            print("Completed step: {0}\n".format(step['step_num']))


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
        assert 'water_content' not in step['WeatheringOutput']
        assert ('time_stamp' in step['WeatheringOutput'])

        print(("Completed step: {0}".format(step['step_num'])))


def test_bulltime():
    '''
    user set time to start emulsification
    This should be in the GnomeOil tests
    '''
    oil = GnomeOil(test_oil)
    assert oil.bullwinkle_time == -999
    oil.bullwinkle_time = 3600
    assert oil.bullwinkle_time == 3600


def test_bullwinkle():
    '''
    user set emulsion constant
    This should be in the GnomeOil tests ...
    '''

    oil = GnomeOil(test_oil)

    # our test_oil is the sample oils
    assert np.isclose(oil.bullwinkle_fraction, 0.1937235)

    oil.bullwinkle_fraction = .4
    assert oil.bullwinkle_fraction == .4


@pytest.mark.skipif(reason="serialization for weatherers overall needs review")
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
