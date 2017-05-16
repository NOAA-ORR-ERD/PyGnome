import gridded
import pytest
from gnome.environment import Environment

# org_new = dict([(c, c.__new__) for c in [Variable, VectorVariable, Time, Grid, Grid_U, Grid_S, Depth]])


class Variable(gridded.Variable, Environment):

    @classmethod
    def from_netCDF(cls, *args, **kwargs):
        return monkeypatch_gridded(super(cls, Variable).from_netCDF, args, kwargs)

    @property
    def time(self):
        return self._time

    @time.setter
    def time(self, t):
        self._time = t


class VectorVariable(gridded.VectorVariable, Environment):

    @classmethod
    def from_netCDF(cls, *args, **kwargs):
        return monkeypatch_gridded(super(cls, VectorVariable).from_netCDF, args, kwargs)

    @property
    def time(self):
        return self._time

    @time.setter
    def time(self, t):
        self._time = t

class Time(gridded.time.Time):
    @classmethod
    def from_netCDF(cls, *args, **kwargs):
        return monkeypatch_gridded(super(cls, Time).from_netCDF, args, kwargs)


class Grid(gridded.grids.Grid):
    pass


class Grid_U(gridded.grids.Grid_U):
    @classmethod
    def from_netCDF(cls, *args, **kwargs):
        return monkeypatch_gridded(super(cls, Grid_U).from_netCDF, args, kwargs)


class Grid_S(gridded.grids.Grid_S):
    @classmethod
    def from_netCDF(cls, *args, **kwargs):
        return monkeypatch_gridded(super(cls, Grid_S).from_netCDF, args, kwargs)


class Depth(gridded.depth.Depth):
    pass

replacements = {gridded.variable.Variable: Variable,
                gridded.variable.VectorVariable: VectorVariable,
                gridded.time.Time: Time,
                gridded.grids.Grid_U: Grid_U,
                gridded.grids.Grid_S: Grid_S,
                gridded.depth.Depth: Depth
                }


def patch__new__(cls, *args, **kwargs):
    newcls = replacements.get(cls, cls)
    return object.__new__(newcls, *args, **kwargs)


def monkeypatch_gridded(func, args, kwargs):
    '''
    Monkeypatches gridded to use the classes within this file, runs the function
    with the args and kwargs, and undoes the patch
    '''
    pytest.set_trace()
    cls_list = [gridded.variable.Variable,
                gridded.variable.VectorVariable,
                gridded.time.Time,
                gridded.grids.Grid_U,
                gridded.grids.Grid_S,
                gridded.depth.Depth]
    orig_new = dict([(kls, kls.__new__) for kls in cls_list])
    for cls in cls_list:
        print 'setting {0}.__new__ to patch__new__'.format(cls)
        cls.__new__ = staticmethod(patch__new__)

    pytest.set_trace()
    rv = func(*args, **kwargs)

    for cls in cls_list:
        print 'resetting {0}.__new__ to {1}'.format(cls, orig_new[cls])
        cls.__new__ = orig_new[cls]

    return rv
