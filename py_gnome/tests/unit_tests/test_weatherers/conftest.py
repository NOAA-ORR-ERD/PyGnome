'''
import all fixtures from ../conftest.py so if user runs tests from this
directory, all fixtures are found
'''
from gnome.environment import constant_wind, Water, Waves

from gnome.weatherers import FayGravityViscous

from ..conftest import test_oil, sample_sc_release

# fixtures used by test_weatherers module - import it here so py.test will find
# when tests are run from test_weatherers/subdirectory
from ..conftest import sample_model_fcn, sample_model_fcn2

from gnome.spills.gnome_oil import GnomeOil
from gnome.ops import weathering_array_types


def weathering_data_arrays(n_arrays,
                           water=None,
                           time_step=15. * 60,
                           substance=None,
                           langmuir=False,
                           num_elements=2,
                           units='g',
                           amount_per_element=1.0):
    '''
    function to initialize data_arrays set by WeatheringData. Weatherer tests
    can use this function to release elements and initialize data without
    defining a model
    '''
    if water is None:
        water = Water(temperature = 300.)
    environment = {'water': water}  
    
    rqd_weatherers = [FayGravityViscous(water)]
    arrays = dict()
    arrays.update(weathering_array_types) #always have base weathering array types available
    arrays.update(n_arrays)
    for wd in rqd_weatherers:
        arrays.update(wd.array_types)

    substance = GnomeOil(test_oil) if substance is None else GnomeOil(substance)
    # if isinstance(str):
    #     substance = GnomeOil(substance)
    # if substance is None:
    #     substance = GnomeOil(test_oil)

    arrays.update(substance.array_types)

    sc = sample_sc_release(num_elements=num_elements,
                           substance=substance,
                           arr_types=arrays,
                           time_step=time_step,
                           units=units,
                           amount_per_element=amount_per_element,
                           environment=environment)
                           
    for wd in rqd_weatherers:
        wd.prepare_for_model_run(sc)
        wd.initialize_data(sc, sc.num_released)

    return (sc, time_step, rqd_weatherers)


def build_waves_obj(wind_speed, wind_units, direction_deg, temperature):
    # also test with lower wind no dispersion
    wind = constant_wind(wind_speed, direction_deg, wind_units)
    water = Water(temperature=temperature)
    waves = Waves(wind, water)

    return waves
