"""
Defines conftest for pytest.
Place configuration options that apply to all tests here, for instance:
* tests in unit_tests/ and integration_tests/ both can access the
  @pytest.mark.slow decorator because pytest_runtest_setup is defined here

* similarly rand.seed(1) is automatically done before every test. Maybe
  worthwhile to find a way to do this only for specific tests

The scope="module" on the fixtures ensures it is only invoked once per
test module
"""
import pytest

from gnome.utilities import rand


def pytest_addoption(parser):
    '''
    Skip slow tests
    '''
    parser.addoption('--runslow',
                     action='store_true',
                     help='run slow tests and all other tests')
    parser.addoption('--serial',
                     action='store_true',
                     help=('run only tests marked as serial. '
                           'used to run tests skipped by xdist'))


def pytest_runtest_setup(item):
    '''
    pytest builtin hook

    This is executed before pytest_runtest_call.
    pytest_runtest_call is invoked to execute the test item.
    So the code in here is executed before each test.
    '''
    if ('slow' in item.keywords and
        not item.config.getoption('--runslow')):
        pytest.skip('need --runslow option to run')

    if (item.config.getoption('--serial') and
        'serial' not in item.keywords):
        pytest.skip('only run tests marked as serial')

    # set random seed:
    # Let's not print anything - it clearly works, its just extra output
    # print "Seed C++, python, numpy random number generator to 1"
    rand.seed(1)
