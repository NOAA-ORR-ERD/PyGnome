'''
update_from_dict has been giving problems on spills so just add a couple of
simple tests for spills, and movers orderedcollection
'''
import datetime as dt
import pytest
import numpy as np

from gnome.environment.gridded_objects_base import Time
from gnome.environment.timeseries_objects_base import (TimeseriesData,
                                                       TimeseriesVector)
from gnome.utilities.serializable_demo_objects import DemoObj

@pytest.fixture('class')
def dates():
    return np.array([dt.datetime(2000, 1, 1, 0),
                     dt.datetime(2000, 1, 1, 2),
                     dt.datetime(2000, 1, 1, 4),
                     dt.datetime(2000, 1, 1, 6),
                     dt.datetime(2000, 1, 1, 8), ])

@pytest.fixture('class')
def series_data():
    return np.array([1,3,6,10,15])

@pytest.fixture('class')
def series_data2():
    return np.array([2,6,12,20,30])

class TestUpdate(object):
    '''
    Uses the DemoObj to test various aspects of updating a gnome object. DemoObj
    has a lot of complicated
    '''
    def prod_inst(self):
        _t = Time(dates())
        tsv = TimeseriesVector(
            variables=[TimeseriesData(name='u', time=_t, data=series_data()),
                       TimeseriesData(name='v', time=_t, data=series_data2())],
            units='m/s'
        )
        filename = ['foo.nc', 'bar.nc']

        inst = DemoObj(filename=filename, variable=tsv, variables=[tsv, tsv.variables[0]])
        return inst

    def test_empty_update(self):
        start = self.prod_inst()
        inst = self.prod_inst()
        inst.update({})
        assert inst == start

    def test_nested_update(self):
        inst = self.prod_inst()
        assert inst.foo_float == 42
        update_dict = {
            'foo_float': 55, #should succeed
            'wrong_foo': 33, #should be ignored
            'variable': {'name': 'updated1'}, #should succeed
            'variables': [{}, {'name':'updated2'}] #should succeed, not affect item 1
        }
        inst.update(update_dict)
        assert inst.variable.name == 'updated1'
        assert inst.variables[0].name == 'updated1'
        assert inst.variables[1].name == 'updated2'
        assert inst.foo_float == 55
        assert not hasattr(inst, 'wrong_foo')
        update_dict2 = {
            'foo_float_array' : [5], #should not be applied (read_only)
            'variables': [{'name': 'changed'}] #should not affect second object
        }
        inst.update(update_dict2)
        assert inst.variable.name == 'changed'
        assert inst.foo_float_array == [42, 84]
        assert inst.variables[0].name == 'changed'
        assert inst.variables[1].name == 'updated2'

        update_dict3 = {
            'variables': [{'variables':[{'name' : 'wow'}]}]
        }
        inst.update(update_dict3)
        assert inst.variable.variables[0].name == 'wow'

    def test_update_none(self):
        inst = self.prod_inst()
        orig = self.prod_inst()
        update_dict = {'name': 'foo', 'variable':None, 'variables': [{}, None]}
        inst.update(update_dict)
        assert inst.name == 'foo'
        assert inst.variable is None
        assert inst.variables[0] == orig.variables[0]
        assert inst.variables[1] is None


