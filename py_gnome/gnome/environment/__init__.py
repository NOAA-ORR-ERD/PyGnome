'''
environment module
'''
from environment import Environment, Conditions
from tide import Tide, TideSchema
from wind import Wind, WindSchema, constant_wind

__all__ = [Environment,
           Conditions,
           Tide,
           TideSchema,
           Wind,
           WindSchema,
           constant_wind]

units = {'gas_constant': 'J/(K mol)',
         'pressure': 'Pa'}
atmos = {'pressure': 101325.0}
constants = {'gas_constant': 8.314}
