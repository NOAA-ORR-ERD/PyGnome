'''
environment module
'''
from environment import Environment, Water, WaterSchema
from tide import Tide, TideSchema
from wind import Wind, WindSchema, constant_wind

__all__ = [Environment,
           Water,
           WaterSchema,
           Tide,
           TideSchema,
           Wind,
           WindSchema,
           constant_wind]

units = {'gas_constant': 'J/(K mol)',
         'pressure': 'Pa'}
constants = {'gas_constant': 8.314,
             'atmos_pressure': 101325.0}
