from datetime import datetime, timedelta
import copy

import pytest
from pytest import raises

import numpy as np
import pdb
from gnome.array_types import default_array_types, gat
from gnome.spill.le import LEData


'''
Tests for LEData container for spills
'''

def test_LEData_init():
    dat = LEData()
    assert dat._array_types == default_array_types
    assert dat._len == 0
    assert dat._bufs == {}

    dat2 = LEData()
    assert dat._array_types is not dat2._array_types
    assert dat._array_types['positions'] is not dat2._array_types['positions']


def test_LEData_prep():
    arrtypes = {'area': gat('area')}

    dat = LEData()
    dat.prepare_for_model_run(arrtypes, 0)
    assert 'area' in dat._array_types
    assert dat['area'].shape == (0,)
    assert dat['positions'].shape == (0,3)
    assert dat._bufs['area'].shape == (100,)
    assert dat._bufs['positions'].shape == (100,3)
    assert dat._len == 0

def test_LEData_rewind():
    dat = LEData()
    dat.prepare_for_model_run({}, 0)
    assert dat['positions'].shape == (0,3)
    assert dat._bufs['positions'].shape == (100,3)
    dat.rewind()
    assert dat._array_types == default_array_types
    assert dat._len == 0
    assert dat._bufs == {}