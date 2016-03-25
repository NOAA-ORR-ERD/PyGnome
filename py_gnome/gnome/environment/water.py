import warnings

import netCDF4 as nc4
import numpy as np

from gnome.utilities.geometry.cy_point_in_polygon import points_in_polys
from datetime import datetime, timedelta
from dateutil import parser
from colander import SchemaNode, Float, MappingSchema, drop, String, OneOf
from gnome.persist.base_schema import ObjType
from gnome.utilities import serializable
from gnome.movers import ProcessSchema

import pyugrid
import pysgrid

class EnvProperty(object):
    def __init__(self,
                 name=None,
                 is_constant=False,
                 constant_vec=None,
                 constant_val=None,
                 constant_direction=None,
                 is_gridded=False,
                 grid=None,
                 data_source=None,
                 time=None,
                 is_timeseries=False,
                 time_series=None
                   **kwargs):
        self.name=name
        self.is_constant=is_constant
        self.constant_val=constant_val
        self.constant_vec=constant_vec
        self.constant_direction=constant_direction
        self.is_gridded=is_gridded
        self.grid = grid
        self.data = data_source
        self.time = time
        self.is_timeseries=is_timeseries
        self.time_series=time_series
    
    @classmethod
    def constant_var(cls, name=None, value=None):
        if name is None or value is None:
            raise ValueError("Name and value must be provided") 
        return cls(name=name, is_constant=True, constant_val=value)
    
    @classmethod
    def gridded_var(cls, name=None, grid=None, data_source=None, time=None):
        if grid is None or data_source is None:
            raise ValueError('Must provide a grid and data source that can fit to the grid')
        if grid.infer_grid(data_source) is None:
            raise ValueError('Data source must be able to fit to the grid')
        return cls(name=name, is_constant=False, grid=grid, data_source=data_source, time=time)

    #Should vector quantities even be supported?
    @classmethod
    def constant_vec(cls, name=None, vector=None, magnitude=None, direction=None):
        if name is None:
            raise ValueError('Name must be provided')
        if vector is None:
            if magnitude is None or direction is None:
                raise ValueError('If vector is not provided, magnitude and direction must be')
            else:
                return cls(name=name, is_constant=True, constant_val=magnitude, constant_direction=direction)
        else:
            vector = np.asarray(vector, dtype=np.double)
            if magnitude is not None or direction is not None:
                warnings.warn("vector is defined, ignoring magnitude and direction")
            if vector.shape != (2,) or vector.shape != (3,):
                raise ValueError("Must provide [u,v] or [u,v,w] for vector")
            return cls(name=name, is_constant=True, constant_vec=vector)
           
    @classmethod
    def gridded_vec(cls, name=None, grid=None, data_source=None):
        if grid is None or data_source is None:
            raise ValueError('Must provide a grid and data source that can fit to the grid')
        if not grid.is_compatible(data_source):
            raise ValueError('Data source must be able to fit to the grid')
        return cls(name=name, is_constant=False, grid=grid, data_source=data_source)

class WaterConditions(object):
    def __init__(self, **kwargs):
        self.temperature_var = kwargs.get('temperature', None)
        self.salinity_var = kwargs.get('salinity', None)
        self.velocity_u, self.velocity_v = kwargs.get('velocity', (None,None))

    @property
    def temperature(self):
        return self.temperature_var
    @temperature.setter
    def temperature(self, variable):
        #TODO
        self.temperature_var=variable
    
    def temperature_at(self, points, time):
        #if single
        if points.shape == (2,) or points.shape == (3,):
            points = points.reshape((-1,points.shape[0]))
            #should we only ever use the x/y coordinates? ingore z until later?
        if temperature_var is None:
            #do not fail catastrophically here?
            warnings.warn("Temperature source is not defined")
            return None
        
        if self.temperature_var.is_constant:
            return np.full((points.shape[0],1),self.temperature_var.constant_val)
        
        if self.temperature_var.is_gridded:
            slices = None
            if self.temperature_var.time is not None:
                slices=[self.temperature_var.time.indexof(time)]
                
            if points.shape[1] == 2:
                return grid.interpolate_var_to_points(points, self.temperature_var.data, slices=slices)
            else:
                return grid.interpolate_var_to_points(points, self.temperature_var.data, slices=slices)
            
        if self.temperature_var.is_time_series:
            return None #TO BE IMPLEMENTED
        
        #if we get here, temperature variable is invalid
        raise RuntimeError("Temperature var is not constant, timeseries, or gridded!")
    
    def salinity_at(self, points):
        pass
    
    def velocity_at(self, points):