



from datetime import datetime, timedelta
import copy

import pytest
from pytest import raises

import numpy as np
import pdb
from gnome.array_types import default_array_types, gat
from gnome.spills.le import LEData


'''
Tests for LEData container for spills
'''

def test_LEData_init():
    dat = LEData()
    assert dat._array_types == default_array_types
    assert dat._bufs == {}

def test_LEData_prep():
    arrtypes = {'area': gat('area')}

    dat = LEData()
    dat.prepare_for_model_run(arrtypes, 0)
    assert 'area' in dat._array_types
    assert dat['area'].shape == (0,)
    assert dat['positions'].shape == (0,3)
    assert dat._bufs['area'].shape == (100,)
    assert dat._bufs['positions'].shape == (100,3)
    assert len(dat) == 0

def test_LEData_extend():
    arrtypes = {'area': gat('area')}

    dat = LEData()
    dat.prepare_for_model_run(arrtypes, 10)
    assert len(dat['area']) == 0
    dat.extend_data_arrays(5)
    assert len(dat['area']) == 5
    assert np.all(dat['area'] == 0)
    dat['area'][:] = 42
    assert np.all(dat['area'] == 42)
    assert len(dat._bufs['area']) == 100

    dat.extend_data_arrays(1000)
    assert dat._bufs['area'].shape == (2010,)
    assert dat._bufs['positions'].shape == (2010,3)
    assert len(dat['area']) == 1005
    assert np.all(dat['area'][0:5] == 42)
    assert np.all(dat['area'][5:] == 0)


def test_LEData_rewind():
    dat = LEData()
    dat.prepare_for_model_run({}, 0)
    assert dat['positions'].shape == (0,3)
    assert dat._bufs['positions'].shape == (100,3)
    dat.rewind()
    assert dat._array_types == default_array_types
    assert dat._bufs == {}
    assert len(dat) == 0