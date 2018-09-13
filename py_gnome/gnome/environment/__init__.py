'''
environment module
'''
from property import EnvProp, VectorProp, Time
from ts_property import TimeSeriesProp, TSVectorProp

from .environment import Environment, env_from_netCDF, ice_env_from_netCDF
from environment_objects import (WindTS,
                                 GridCurrent,
                                 GridWind,
                                 IceVelocity,
                                 IceConcentration,
                                 GridTemperature,
                                 IceAwareCurrent,
                                 IceAwareWind,
                                 TemperatureTS)

from .water import Water, WaterSchema
from .waves import Waves, WavesSchema
from .tide import Tide, TideSchema
from .wind import Wind, WindSchema, constant_wind, wind_from_values

from running_average import RunningAverage, RunningAverageSchema
from gridded_objects_base import PyGrid, GridSchema
from grid import Grid
# from gnome.environment.environment_objects import IceAwareCurrentSchema


__all__ = [Environment,
           Water,
           WaterSchema,
           Waves,
           WavesSchema,
           Tide,
           TideSchema,
           Wind,
           WindSchema,
           RunningAverage,
           RunningAverageSchema,
           PyGrid,
           GridSchema,
           constant_wind,
           WindTS,
           GridCurrent,
           GridWind,
           IceConcentration,
           IceVelocity,
           GridTemperature,
           IceAwareCurrent,
           # IceAwareCurrentSchema,
           IceAwareWind,
           TemperatureTS,
           env_from_netCDF,
           ice_env_from_netCDF
           ]
