'''
import all fixtures from ../conftest.py so if user runs tests from this
directory, all fixtures are found
'''
from gnome.environment import Water
from gnome.weatherers import WeatheringData, FayGravityViscous
from gnome.spill.elements import floating

from ..conftest import test_oil, sample_sc_release

# fixtures used by test_weatherers module - import it here so py.test will find
# when tests are run from test_weatherers/subdirectory
from ..conftest import sample_model_fcn, sample_model_fcn2


def weathering_data_arrays(n_arrays,
                           water=None,
                           time_step=15.*60,
                           element_type=None,
                           langmuir=False,
                           num_elements=2):
    '''
    function to initialize data_arrays set by WeatheringData. Weatherer tests
    can use this function to release elements and initialize data without
    defining a model
    '''
    if water is None:
        water = Water()
    rqd_weatherers = [WeatheringData(water), FayGravityViscous(water)]
    arrays = set()
    arrays.update(n_arrays)
    for wd in rqd_weatherers:
        arrays.update(wd.array_types)

    if element_type is None:
        element_type = floating(substance=test_oil)

    sc = sample_sc_release(num_elements=num_elements,
                           element_type=element_type,
                           arr_types=arrays,
                           time_step=time_step)
    for wd in rqd_weatherers:
        wd.prepare_for_model_run(sc)
        wd.initialize_data(sc, sc.num_released)

    return (sc, time_step, rqd_weatherers)
