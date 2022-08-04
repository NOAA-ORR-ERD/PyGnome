'''
Test evaporation module
'''





from datetime import timedelta, datetime

import pytest
import numpy as np

from gnome.model import Model
from gnome.spills import surface_point_line_spill
from gnome.environment import constant_wind, Water, Wind
from gnome.weatherers import Evaporation
from gnome.outputters import WeatheringOutput
from gnome.basic_types import oil_status

from .conftest import weathering_data_arrays, test_oil
from ..conftest import (# sample_model,
                        sample_model_weathering)


def test_evaporation_no_wind():
    evap = Evaporation(Water(), wind=constant_wind(0., 0))
    (sc, time_step) = weathering_data_arrays(evap.array_types, evap.water)[:2]

    model_time = (sc.spills[0].release_time +
                  timedelta(seconds=time_step))

    evap.prepare_for_model_run(sc)
    evap.prepare_for_model_step(sc, time_step, model_time)
    evap.weather_elements(sc, time_step, model_time)
    for spill in sc.spills:
        mask = sc.get_spill_mask(spill)
        assert np.all(sc['evap_decay_constant'][mask, :] < 0.0)


@pytest.mark.parametrize(('oil', 'temp', 'num_elems', 'on'),
                         [(test_oil, 311.15, 3, True),
                          ('oil_6', 311.15, 3, False)
                          ])
def test_evaporation(oil, temp, num_elems, on):
    '''
    still working on tests ..
    '''
    time_step = 15. * 60

    evap = Evaporation(Water(), wind=constant_wind(1., 0))
    evap.on = on

    sc = weathering_data_arrays(evap.array_types, evap.water, time_step)[0]

    model_time = (sc.spills[0].release_time + timedelta(seconds=time_step))

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
        if on:
            assert np.all(sc['evap_decay_constant'][mask, :] < 0.0)
        else:
            assert np.all(sc['evap_decay_constant'][mask, :] == 0.0)

    print('\nevap_decay_const', sc['evap_decay_constant'])
    print('frac_lost', sc['frac_lost'])

    if on:
        assert sc.mass_balance['evaporated'] > 0.0
        print('total evaporated', sc.mass_balance['evaporated'])
    else:
        assert 'evaporated' not in sc.mass_balance
        assert np.all(sc['mass_components'] == init_mass)


class TestDecayConst(object):
    '''
    WIP - Currently has one working test, but may have more so grouped it in
    a class
    '''
    def setup_test(self, end_time_delay, num_les, ts=900.):
        stime = datetime(2015, 1, 1, 12, 0)
        etime = stime + end_time_delay
        st_pos = (0, 0, 0)
        oil = test_oil

        m1 = Model(start_time=stime, time_step=ts)
        m1.environment += [constant_wind(0, 0), Water()]
        m1.weatherers += Evaporation()
        m1.spills += surface_point_line_spill(num_les[0], st_pos, stime,
                                              end_release_time=etime,
                                              substance=oil,
                                              amount=36000, units='kg')
        m1.outputters += WeatheringOutput()

        m2 = Model(start_time=stime, time_step=ts)
        m2.environment += [constant_wind(0, 0), Water()]
        m2.weatherers += Evaporation()
        m2.spills += surface_point_line_spill(num_les[1], st_pos, stime,
                                              end_release_time=etime,
                                              substance=oil,
                                              amount=36000, units='kg')
        m2.outputters += WeatheringOutput()
        return (m1, m2)

    @pytest.mark.skipif(reason="not sure how to test dependence on timestep")
    def test_evap_decay_const_vary_ts(self, delay=timedelta(0)):
        '''
        evap decay constant does depend on timestep since thickness has a
        nonlinear dependence on age so varying the timestep gives different
        evaporation results
        '''
        (m1, m2) = self.setup_test(delay, (10, 10))
        m2.time_step = 900

        for ix in range(m1.num_time_steps):
            w1 = m1.step()['WeatheringOutput']
            if ix == 0:
                w2 = m2.step()['WeatheringOutput']

            if ix > 0:
                for _ in range(4):
                    w2 = m2.step()['WeatheringOutput']

                val1 = list(w1.values())
                val2 = list(w2.values())
                d_time1 = val1.pop(4)
                d_time2 = val2.pop(4)

                if d_time1 == d_time2:
                    assert np.allclose(val1, val2)

    @pytest.mark.parametrize("end_time_delay", [timedelta(hours=0),
                                                timedelta(hours=4)])
    def test_evap_decay_const_vary_numLE(self, end_time_delay):
        '''
        test checks the evaporation decay constant does not depend on the
        number of elements.
        '''
        # for a 15min timestep, make sure at least one LE per timestep is
        # released for test to work.
        if end_time_delay.total_seconds() == 0:
            num_les_one_per_ts = 1
        else:
            num_les_one_per_ts = end_time_delay.total_seconds() / 900.

        (m1, m2) = self.setup_test(end_time_delay, (2*num_les_one_per_ts,
                                                    4 * num_les_one_per_ts))

        for ix in range(m1.num_time_steps):
            w1 = m1.step()['WeatheringOutput']
            w2 = m2.step()['WeatheringOutput']

            d_time1 = w1.pop('time_stamp')
            d_time2 = w2.pop('time_stamp')

            print("Completed step ", ix)
            assert d_time1 == d_time2
            assert np.allclose(list(w1.values()), list(w2.values()))


def assert_helper(sc, new_p):
    'common assertions for spills and data in SpillContainer'
    total_mass = sum([spill.get_mass() for spill in sc.spills])
    arrays = {'evap_decay_constant', 'mass_components', 'mass', 'status_codes'}

    substances_list = sc.itersubstancedata(arrays)
    print(substances_list)
    for substance, data in substances_list:
        if len(sc) > new_p:
            old_le = len(sc) - new_p
            inwater = data['status_codes'][:old_le] == oil_status.in_water
            assert np.all(data['evap_decay_constant'][:old_le, :][inwater] <
                          0.0)
            assert np.all(data['evap_decay_constant'][:old_le, :][~inwater] ==
                          0.0)

            assert np.allclose(np.sum(data['mass_components'], 1),
                               data['mass'])

            # not an instantaneous release so following is true even at step 0
            assert data['mass'].sum() < total_mass

        if new_p > 0:
            #new_p/2 because of new step release behavior May 2020
            assert np.all(data['evap_decay_constant'][ - (new_p // 2):, :] == 0.0)


@pytest.mark.parametrize(('oil', 'temp'), [('oil_6', 333.0),
                                           (test_oil, 311.15),
                                           ])
def test_full_run(sample_model_fcn, oil, temp):
    '''
    test evaporation outputs for a full run of model.
    This contains a mover so at some point several elements end up on_land.
    This test also checks the evap_decay_constant for elements that are not
    in water is 0 so mass is unchanged.
    '''
    model = sample_model_weathering(sample_model_fcn, oil, temp, 10)
    model.environment += [Water(temp), constant_wind(1., 0)]
    model.weatherers += [Evaporation(model.environment[-2],
                                     model.environment[-1])]
    released = 0
    init_rho = model.spills[0].substance.density_at_temp(temp)
    init_vis = model.spills[0].substance.kvis_at_temp(temp)
    for step in model:
        for sc in list(model.spills.items()):
            assert_helper(sc, sc.num_released - released)
            released = sc.num_released
            if sc.num_released > 0:
                assert np.all(sc['density'] >= init_rho)
                assert np.all(sc['viscosity'] >= init_vis)

            mask = sc['status_codes'] == oil_status.in_water
            assert sc.mass_balance['floating'] == np.sum(sc['mass'][mask])

            print(("Amount released: {0}".
                   format(sc.mass_balance['amount_released'])))
            print("Mass floating: {0}".format(sc.mass_balance['floating']))
            print("Mass evap: {0}".format(sc.mass_balance['evaporated']))
            print("LEs in water: {0}".format(sum(mask)))
            print("Mass on land: {0}".format(np.sum(sc['mass'][~mask])))

            print("Completed step: {0}\n".format(step['step_num']))

    # print("failing on purpose")
    # assert False


def test_full_run_evap_not_active(sample_model_fcn):
    'no water/wind object'
    model = sample_model_weathering(sample_model_fcn, 'oil_6')
    model.weatherers += Evaporation(on=False)
    model.outputters += WeatheringOutput()
    for step in model:
        '''
        if no weatherers, then no weathering output - need to add on/off
        switch to WeatheringOutput
        '''
        assert 'evaporated' not in step['WeatheringOutput']
        assert ('time_stamp' in step['WeatheringOutput'])

        print(("Completed step: {0}".format(step['step_num'])))


@pytest.mark.skipif(reason="serialization for weatherers overall needs review")
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
