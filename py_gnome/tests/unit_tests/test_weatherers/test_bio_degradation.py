'''
Test biodegradation module
'''






from datetime import timedelta

import pytest
import numpy as np

from gnome.environment import constant_wind, Water, Waves
# from gnome.spills.elements import floating

from gnome.weatherers import (Evaporation,
                              NaturalDispersion,
                              # Dissolution,
                              Biodegradation,
                              weatherer_sort)

from .conftest import weathering_data_arrays, test_oil


from ..conftest import (sample_model_weathering2)

from pprint import PrettyPrinter

pp = PrettyPrinter(indent=2, width=120)

wind = constant_wind(15., 270, 'knots')
water = Water()
waves = Waves(wind, water)


def test_init():
    wind = constant_wind(15., 0)
    waves = Waves(wind, Water())
    bio_deg = Biodegradation(waves)

    print(bio_deg.array_types)
    assert all([(at in bio_deg.array_types)
                for at in ('mass', 'droplet_avg_size')])


def test_sort_order():
    'test sort order for Biodegradation weatherer'
    wind = constant_wind(15., 0)
    waves = Waves(wind, Water())
    bio_deg = Biodegradation(waves)

    assert weatherer_sort(bio_deg) == 11


@pytest.mark.skipif(reason="serialization for weatherers overall needs review")
def test_serialize_deseriailize():
    'test serialize/deserialize for webapi'

    wind = constant_wind(15., 0)
    water = Water()
    waves = Waves(wind, water)

    bio_deg = Biodegradation(waves)
    json_ = bio_deg.serialize()
    pp.pprint(json_)

    assert json_['waves'] == waves.serialize()

    # deserialize and ensure the dict's are correct
    d_ = Biodegradation.deserialize(json_)
    assert d_['waves'] == Waves.deserialize(json_['waves'])

    d_['waves'] = waves
    bio_deg.update_from_dict(d_)

    assert bio_deg.waves is waves


def test_prepare_for_model_run():

    bio_deg = Biodegradation(waves)

    (sc, time_step) = weathering_data_arrays(bio_deg.array_types,
                                             water)[:2]

    assert 'bio_degradation' not in sc.mass_balance

    bio_deg.prepare_for_model_run(sc)

    assert 'bio_degradation' in sc.mass_balance


@pytest.mark.skipif(reason="refactoring sara out of gnome oil")
@pytest.mark.parametrize(('oil', 'temp', 'num_elems', 'expected_mb', 'on'),
                         [('oil_ans_mp', 311.15, 3, 0.0, True),
                          #('ABU SAFAH', 311.15, 3, 0.0, True),
                          ('oil_bahia', 311.15, 3, 0.0, True),
                          ('oil_ans_mp', 311.15, 3, np.nan, False)])
def test_bio_degradation_mass_balance(oil, temp, num_elems, expected_mb, on):

    bio_deg = Biodegradation(waves)
    (sc, time_step) = weathering_data_arrays(bio_deg.array_types,
                                             water,
                                             num_elements=num_elems)[:2]
    model_time = (sc.spills[0].release_time +
                  timedelta(seconds=time_step))

    bio_deg.on = on
    bio_deg.prepare_for_model_run(sc)
    bio_deg.initialize_data(sc, sc.num_released)

    bio_deg.prepare_for_model_step(sc, time_step, model_time)
    bio_deg.weather_elements(sc, time_step, model_time)

    if on:
        assert np.isclose(sc.mass_balance['bio_degradation'], expected_mb)
    else:
        assert 'bio_degradation' not in sc.mass_balance


@pytest.mark.skipif(reason="refactoring sara out of gnome oil")
@pytest.mark.parametrize(('oil', 'temp', 'expected_balance'),
    # TODO - expected ballance values:
                         [(test_oil, 288.7, -1),
                          #('ABU SAFAH', 288.7, -1),
                          ('oil_ans_mp', 288.7, -1),
                          #('ALASKA NORTH SLOPE, OIL & GAS', 279.261,
                           #-1),
                          ('oil_bahia', 288.7, -1)])
def test_bio_degradation_full_run(sample_model_fcn2,
                                  oil, temp, expected_balance):
    '''
    test bio degradation outputs post step for a full run of model. Dump json
    for 'weathering_model.json' in dump directory
    '''
    model = sample_model_weathering2(sample_model_fcn2, oil, temp)
    # model.duration = timedelta(days=5)
    model.environment += [Water(temp), wind, waves]
    model.weatherers += Evaporation()
    model.weatherers += NaturalDispersion()
    # model.weatherers += Dissolution(waves)
    model.weatherers += Biodegradation()

    for sc in model.spills.items():
        print(sc.__dict__.keys())
        print(sc._data_arrays)

        print('num spills:', len(sc.spills))
        print('spill[0] amount:', sc.spills[0].amount)
        original_amount = sc.spills[0].amount

    # set make_default_refs to True for objects contained in model after adding
    # objects to the model
    model.set_make_default_refs(True)
    model.setup_model_run()

    bio_degradated = []
    for step in model:
        for sc in model.spills.items():
            if step['step_num'] > 0:
                assert (sc.mass_balance['bio_degradation'] > 0)
            if 'bio_degradation' in sc.mass_balance:
                bio_degradated.append(sc.mass_balance['bio_degradation'])

    print('Bio degradated amount: {}'
          .format(bio_degradated[-1]))

    print('Fraction bio degradated after full run: {}'
          .format(bio_degradated[-1] / original_amount))

    assert bio_degradated[0] == 0.0

    # assert np.isclose(bio_degradated[-1], expected_balance)