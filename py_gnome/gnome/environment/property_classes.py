import warnings
import copy

import netCDF4 as nc4
import numpy as np

from datetime import datetime, timedelta
from colander import SchemaNode, Float, Boolean, Sequence, MappingSchema, drop, String, OneOf, SequenceSchema, TupleSchema, DateTime
from gnome.persist.base_schema import ObjType
from gnome.utilities import serializable
from gnome.movers import ProcessSchema
from gnome.persist import base_schema

import pyugrid
import pysgrid
import unit_conversion
from gnome.environment.property import TSVectorProp, GridVectorProp, TimeSeriesProp, GriddedProp

class PropertySchema(base_schema.ObjType):
    name = SchemaNode(String(), missing='default')
    units = SchemaNode(typ=Sequence(accept_scalar=True), children=[SchemaNode(String(), missing=drop),SchemaNode(String(), missing=drop)])
    time = SequenceSchema(SchemaNode(DateTime(default_tzinfo=None), missing=drop), missing=drop)
    extrapolate = SchemaNode(Boolean(), missing='False')
    varnames = SequenceSchema(SchemaNode(String(), missing=drop))


class VelocityTSSchema(PropertySchema):
    timeseries = SequenceSchema(
                                TupleSchema(
                                            children=[SchemaNode(DateTime(default_tzinfo=None), missing=drop),
                                                      TupleSchema(children=[
                                                                            SchemaNode(Float(), missing=0),
                                                                            SchemaNode(Float(), missing=0)
                                                                            ]
                                                                 )
                                                      ],
                                            missing=drop)
                                )


class VelocityTS(TSVectorProp, serializable.Serializable):

    _state = copy.deepcopy(serializable.Serializable._state)
    _schema = VelocityTSSchema

    _state.add_field([serializable.Field('units', save=True, update=True),
                      serializable.Field('timeseries', save=True, update=True),
                      serializable.Field('varnames', save=True, update=True),
                      serializable.Field('extrapolate', save=True, update=True)])

    def __init__(self, name=None, units=None, time = None, components = None, extrapolate=False):
        TSVectorProp.__init__(self, name, units, time=time, variables=components, extrapolate=extrapolate)


    def __repr__(self):
        return self.serialize(json_='save').__repr__()

    @property
    def timeseries(self):
        x = self.variables[0].data
        y = self.variables[1].data
        return map(lambda t,x,y:(t,(x,y)), self._time, x, y)

    def set_data(self, times, datas, names, units):
        for i, c in enumerate(self.components):
            if c.name in names:
                c.set_ts(t, data[i])

    def serialize(self, json_='webapi'):
        dict_ = serializable.Serializable.serialize(self, json_=json_)
        #The following code is to cover the needs of webapi
        if json_ == 'webapi':
            dict_.pop('timeseries')
            dict_.pop('units')
            x = self.variables[0].data
            y = self.variables[1].data
            direction = -(np.arctan2(y,x)*180/np.pi + 90)
            magnitude = np.sqrt(x**2 + y**2)
            ts = (unicode(tx.isoformat()) for tx in self._time)
            dict_['timeseries'] = map(lambda t,x,y:(t,(x,y)), ts, magnitude, direction)
            dict_['units'] = (unicode(self.variables[0].units), u'degrees')
            dict_['varnames'] = [u'magnitude', u'direction', dict_['varnames'][0], dict_['varnames'][1]]
        return dict_


    @classmethod
    def deserialize(cls, json_):
        dict_ = super(VelocityTS, cls).deserialize(json_)

        ts, data = zip(*dict_.pop('timeseries'))
        ts = np.array(ts)
        data = np.array(data).T
        units = dict_['units']
        if len(units) > 1 and units[1] == 'degrees':
            u_data, v_data = data
            v_data = ((-v_data - 90) * np.pi/180)
            u_t = u_data *np.cos(v_data)
            v_data = u_data * np.sin(v_data)
            u_data = u_t
            data = np.array((u_data, v_data))
            dict_['varnames'] = dict_['varnames'][2:]

        units = units[0]
        dict_['units'] = units
        dict_['time'] = ts
        dict_['data'] = data
        return dict_

    @classmethod
    def new_from_dict(cls, dict_):
        varnames = dict_.pop('varnames')
        extrapolate = dict_['extrapolate']
        vars = []
        for i, varname in enumerate(varnames):
            vars.append(TimeSeriesProp(name= varname,
                                       units= dict_['units'],
                                       time = dict_['time'],
                                       data = dict_['data'][i],
                                       extrapolate = dict_['extrapolate']))
        dict_.pop('data')
        dict_['components'] = vars
        return super(VelocityTS, cls).new_from_dict(dict_)

    def update_from_dict(self, data):
        list_ = self._state.get_names('update')
        updated = False

        vns = data.pop('varnames')
        dat = data.pop('data')
        t = data.pop('time')
        u = data.pop('units')
        ex = data.pop('extrapolate')
        for i, c in enumerate(self.components):
            c.set_ts(name = vns[i],
                     units = u[i],
                     data = dat[i],
                     time = t,
                     extrapolate = ex)
        updated = True

        return updated


class VelocityGridSchema(PropertySchema):
    data_file = SchemaNode(String(), missing=drop)
    grid_file = SchemaNode(String(), missing=drop)


class VelocityGrid(GridVectorProp):
    _state = copy.deepcopy(serializable.Serializable._state)

    _schema = VelocityGridSchema
    _state.add([serializable.Field('units', save=True, update=True),
                serializable.Field('varnames', save=True, update=True),
                serializable.Field('extrapolate', save=True, update=True),
                serializable.Field('time', save=True, update=True),
                serializable.Field('data_file', save=True, update=True),
                serializable.Field('grid_file', save=True, update=True)])

if __name__ == "__main__":
    import datetime as dt
    dates = np.array([dt.datetime(2000, 1, 1, 0), dt.datetime(2000, 1, 1, 2), dt.datetime(2000, 1, 1, 4)])
    u_data = np.array([3, 4, 5])
    v_data = np.array([4, 3, 12])
    u = TimeSeriesProp('u', 'm/s', dates, u_data)
    v = TimeSeriesProp('v', 'm/s', dates, v_data)

    print u.at(np.array([(1, 1), (1, 2)]), dt.datetime(2000, 1, 1, 1))

    vprop = TSVectorProp('velocity', 'm/s', variables=[u, v])
    print vprop.at(np.array([(1, 1), (1, 2)]), dt.datetime(2000, 1, 1, 3))

    vel = VelocityTS('test_vel', components = [u,v])
    print vel.at(np.array([(1, 1), (1, 2)]), dt.datetime(2000, 1, 1, 3))

    import pprint
    pp = pprint.PrettyPrinter(indent=4)
    pp.pprint(vel.serialize())
    pp.pprint(VelocityTS.deserialize(vel.serialize()))

    velfromweb = VelocityTS.new_from_dict(VelocityTS.deserialize(vel.serialize()))
    velfromweb.name = 'velfromweb'

    pp.pprint(vel.serialize(json_='save'))
    pp.pprint(velfromweb.serialize(json_='save'))

    velfromsave = VelocityTS.new_from_dict(VelocityTS.deserialize(velfromweb.serialize(json_='save')))
    pp.pprint(velfromsave)

    velfromsave.at(np.array([(0,0)]), datetime(2000,1,1,0,0))

    url = ('http://geoport.whoi.edu/thredds/dodsC/clay/usgs/users/jcwarner/Projects/Sandy/triple_nest/00_dir_NYB05.ncml')
    test_grid = pysgrid.load_grid(url)
    grid_u = test_grid.u
    grid_v = test_grid.v
    grid_time = test_grid.ocean_time._data

    u2 = GriddedProp('u','m/s', time=grid_time, data=grid_u, grid=test_grid, data_file=url, grid_file=url)
    v2 = GriddedProp('v','m/s', time=grid_time, data=grid_v, grid=test_grid, data_file=url, grid_file=url)

    print "got here"
    vel2 = Velocity(name='gridvel', components=[u2, v2])

#     pp.pprint(vel2.serialize())

    pass
