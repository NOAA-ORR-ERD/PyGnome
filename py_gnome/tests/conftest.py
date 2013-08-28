"""
Defines conftest for pytest.
Place configuration options that apply to all tests here, for instance:
* tests in unit_tests/ and integration_tests/ both can access the 
  @pytest.mark.slow decorator because pytest_runtest_setup is defined here
  
* similarly rand.seed(1) is automatically done before every test. Maybe 
  worthwhile to find a way to do this only for specific tests

The scope="module" on the fixtures ensures it is only invoked once per test module
"""
import sys, os
from datetime import datetime

import numpy as np
import pytest

from gnome import basic_types
from gnome.utilities import rand

def pytest_addoption(parser):
    '''
    Skip slow tests
    '''
    parser.addoption("--runslow", action="store_true",
        help="run slow tests")

def pytest_runtest_setup(item):
    """
    pytest builtin hook
    
    This is executed before pytest_runtest_call. 
    pytest_runtest_call is invoked to execute the test item. So the code in here
    is executed before each test.
    """
    if 'slow' in item.keywords and not item.config.getoption("--runslow"):
        pytest.skip("need --runslow option to run")
        
    # set random seed:
    # Let's not print anything - it clearly works, its just extra output
    #print "Seed C++, python, numpy random number generator to 1"
    rand.seed(1)
