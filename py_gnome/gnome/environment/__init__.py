'''
environment module
'''
# ruff: noqa: I001  -- import order is load-order constrained: running_average imports Wind from this module
from .environment import Environment, env_from_netCDF, ice_env_from_netCDF
from .environment_objects import (
    FileGridCurrent as FileGridCurrent,
    GridCurrent,
    GridTemperature as GridTemperature,
    GridWind,
    IceAwareCurrent,
    IceAwareWind,
    IceConcentration,
    IceVelocity as IceVelocity,
    SteadyUniformCurrent,
    TemperatureTS as TemperatureTS,
    WindTS as WindTS,
)
from .gridcur import from_gridcur as from_gridcur
from .water import Water
from .water import WaterSchema as WaterSchema
from .waves import Waves
from .waves import WavesSchema as WavesSchema
from .tide import Tide
from .tide import TideSchema as TideSchema
from .wind import Wind, constant_wind, wind_from_values
from .wind import WindSchema as WindSchema
from .running_average import RunningAverage
from .running_average import RunningAverageSchema as RunningAverageSchema
from .timeseries_objects_base import (
    TimeseriesData,
    TimeseriesVector,
)
from .timeseries_objects_base import (
    TimeseriesDataSchema as TimeseriesDataSchema,
)
from .timeseries_objects_base import (
    TimeseriesVectorSchema as TimeseriesVectorSchema,
)
from .gridded_objects_base import (
    PyGrid,
    Variable,
    VectorVariable,
)
from .gridded_objects_base import (
    GridSchema as GridSchema,
)
from .grid import Grid as Grid
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

# These are the operational environment objects
env_objs = [Water,
            Waves,
            Tide,
            Wind,
            RunningAverage,
            GridCurrent,
            SteadyUniformCurrent,
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

# This hack is for backwards compat on save files...should probably
# remove at some point
import sys

if ('gnome.environment.ts_property' not in sys.modules):
    sys.modules['gnome.environment.ts_property'] = timeseries_objects_base
ts_property = timeseries_objects_base

# __ all__ is used by autoAPI to decide what to document
#  we don't want all these import documented here, as they are already documented elsewhere
# otherwise, it's used for "import *", which we really don't need to support.
__all__ = []
# __all__ = [cls.__name__ for cls in base_classes]
# __all__.extend([cls.__name__ for cls in env_objs])
