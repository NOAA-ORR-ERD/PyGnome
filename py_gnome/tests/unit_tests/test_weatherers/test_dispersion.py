'''
Test natural dispersion module
'''
import os
import json
from datetime import timedelta

import pytest
import numpy as np

from gnome.environment import constant_wind, Water, Waves
from gnome.weatherers import (NaturalDispersion,
                              Evaporation,
                              Emulsification,
                              WeatheringData)
from gnome.outputters import WeatheringOutput
from gnome.spill.elements import floating

from ..conftest import (sample_sc_release,
                        sample_model_weathering,
                        sample_model_weathering2)


#pytestmark = pytest.mark.skipif()

water = Water()
wind = constant_wind(15., 270, 'knots')	#also test with lower wind no dispersion
waves = Waves(wind,water)

arrays = NaturalDispersion().array_types
intrinsic = WeatheringData(water)
arrays.update(intrinsic.array_types)

@pytest.mark.parametrize(('oil', 'temp', 'num_elems', 'on'),
                          [('ABU SAFAH', 311.15, 3, True),
                           ('BAHIA', 311.15, 3, True),
                           ('ALASKA NORTH SLOPE (MIDDLE PIPELINE)', 311.15, 3, False)])
def test_dispersion(oil, temp, num_elems, on):
     '''
     Fuel Oil #6 does not exist...
     '''
     et = floating(substance=oil)
     sc = sample_sc_release(num_elements=num_elems,
                            element_type=et,
                            arr_types=arrays)
     sc.amount = 10000
     time_step = 15. * 60
     intrinsic.update(sc.num_released, sc)
     model_time = (sc.spills[0].get('release_time') +
                   timedelta(seconds=time_step))
 
     disp = NaturalDispersion(waves, water)
     disp.on = on
 
     disp.prepare_for_model_run(sc)
 
     print "oil"
     print oil
     disp.prepare_for_model_step(sc, time_step, model_time)
     disp.weather_elements(sc, time_step, model_time)
 
     if on:
         assert sc.weathering_data['natural_dispersion'] > 0
         assert sc.weathering_data['sedimentation'] > 0
         print "sc.weathering_data['natural_dispersion']"
         print sc.weathering_data['natural_dispersion']
         print "sc.weathering_data['sedimentation']"
         print sc.weathering_data['sedimentation']
     else:
         #assert np.all(sc.weathering_data['natural_dispersion'] == 0)
         assert 'natural_dispersion' not in sc.weathering_data
         assert 'sedimentation' not in sc.weathering_data
 

@pytest.mark.parametrize(('oil', 'temp', 'num_elems'),
                          [('ABU SAFAH', 288.15, 3)])
def test_dispersion_not_active(oil, temp, num_elems):
     '''
     Fuel Oil #6 does not exist...
     '''
     et = floating(substance=oil)
     sc = sample_sc_release(num_elements=num_elems,
                            element_type=et,
                            arr_types=arrays)
     sc.amount = 10000
     time_step = 15. * 60
     intrinsic.update(sc.num_released, sc)
     model_time = (sc.spills[0].get('release_time') +
                   timedelta(seconds=time_step))
 
     disp = NaturalDispersion(waves, water)
 
     disp.prepare_for_model_run(sc)
  
     new_model_time = (sc.spills[0].get('release_time') +
                   timedelta(seconds=3600))
 
     disp.active_start = new_model_time
     disp.prepare_for_model_step(sc, time_step, model_time)
     disp.weather_elements(sc, time_step, model_time)

     assert np.all(sc.weathering_data['natural_dispersion'] == 0)
     assert np.all(sc.weathering_data['sedimentation'] == 0)
     #print "sc.weathering_data['natural_dispersion']"
     #print sc.weathering_data['natural_dispersion']

@pytest.mark.parametrize(('oil', 'temp'), [('ABU SAFAH', 288.7),
                                            ('ALASKA NORTH SLOPE (MIDDLE PIPELINE)', 288.7),
                                            ('BAHIA', 288.7),
                                             ])
def test_full_run(sample_model_fcn2, oil, temp):
    '''
    test dispersion outputs post step for a full run of model. Dump json
    for 'weathering_model.json' in dump directory
    '''
    model = sample_model_weathering2(sample_model_fcn2, oil, temp)
    model.water = Water(temp)
    model.environment += [wind,  waves]
    model.weatherers += Evaporation(model.water, wind)
    model.weatherers += Emulsification(waves)
    model.weatherers += NaturalDispersion(waves, model.water)

    for step in model:
        for sc in model.spills.items():
            if step['step_num'] > 0:
                assert (sc.weathering_data['natural_dispersion'] > 0)
                assert (sc.weathering_data['sedimentation'] > 0)
            print ("Dispersed: {0}".
                   format(sc.weathering_data['natural_dispersion']))
            print ("Sedimentation: {0}".
                   format(sc.weathering_data['sedimentation']))
            print "Completed step: {0}\n".format(step['step_num'])
 

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
          assert len(step['WeatheringOutput']) == 2
          assert ('step_num' in step['WeatheringOutput'] and
                  'time_stamp' in step['WeatheringOutput'])
          #print ("Completed step: {0}"
                 #.format(step['WeatheringOutput']['step_num']))
  
 
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
