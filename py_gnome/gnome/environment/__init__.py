'''
environment module
'''
from environment import Environment
from tide import Tide, TideSchema
from wind import Wind, WindSchema, constant_wind

__all__ = [Environment, Tide, TideSchema, Wind, WindSchema, constant_wind]

# define global environmental properties in SI units
units = {'temperature': 'K',
         'gas_constant': 'J/(K mol)',
         'pressure': 'Pa'}
water = {'temperature': 311.15}
atmos = {'pressure': 101325.0}
constants = {'gas_constant': 8.314}
