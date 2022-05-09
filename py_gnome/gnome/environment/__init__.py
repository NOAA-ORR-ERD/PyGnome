'''
environment module
'''

from .environment import Environment, env_from_netCDF, ice_env_from_netCDF
from .environment_objects import (WindTS,
                                  GridCurrent,
                                  GridWind,
                                  IceVelocity,
                                  IceConcentration,
                                  GridTemperature,
                                  IceAwareCurrent,
                                  IceAwareWind,
                                  TemperatureTS,
                                  FileGridCurrent,
                                  )
from .gridcur import from_gridcur
from .water import Water, WaterSchema
from .waves import Waves, WavesSchema
from .tide import Tide, TideSchema
from .wind import Wind, WindSchema, constant_wind, wind_from_values

from .running_average import RunningAverage, RunningAverageSchema
from .timeseries_objects_base import (TimeseriesData,
                                     TimeseriesDataSchema,
                                     TimeseriesVector,
                                     TimeseriesVectorSchema
                                     )
from .gridded_objects_base import (PyGrid,
                                  GridSchema,
                                  VectorVariable,
                                  Variable)

from .grid import Grid

from . import timeseries_objects_base

# from gnome.environment.environment_objects import IceAwareCurrentSchema

base_classes = [Environment,
                PyGrid,
                Variable,
                VectorVariable,
                TimeseriesData,
                TimeseriesVector]

helper_functions = [env_from_netCDF,
                    ice_env_from_netCDF,
                    constant_wind,
                    wind_from_values,
                    ]

#These are the operational environment objects
env_objs = [Water,
            Waves,
            Tide,
            Wind,
            RunningAverage,
            GridCurrent,
            GridWind,
            IceConcentration,
            IceAwareCurrent,
            IceAwareWind]

# schemas = set()
# for cls in env_objs:
#     if hasattr(cls, '_schema'):
#         schemas.add(cls._schema)
# schemas = list(schemas)
schemas = list({cls._schema for cls in env_objs if hasattr(cls, '_schema')})

#This hack is for backwards compat on save files...should probably
#remove at some point
import sys
if ('gnome.environment.ts_property' not in sys.modules):
    sys.modules['gnome.environment.ts_property'] = timeseries_objects_base
ts_property = timeseries_objects_base

__all__ = [cls.__name__ for cls in base_classes]
__all__.extend([cls.__name__ for cls in env_objs])
