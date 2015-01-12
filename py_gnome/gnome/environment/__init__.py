'''
environment module
'''
from environment import Environment, Water, WaterSchema
from waves import Waves, WavesSchema
from tide import Tide, TideSchema
from wind import Wind, WindSchema, constant_wind
#from running_average import RunningAverage, RunningAverageSchema

__all__ = [Environment,
           Water,
           WaterSchema,
           Waves,
           WavesSchema,
           Tide,
           TideSchema,
           Wind,
           WindSchema,
           #RunningAverage,
           #RunningAverageSchema,
           constant_wind]

'''
Constants are mostly used internally. They are all documented here to keep them
in one place. The 'units' serve as documentation ** do not mess with them **.
There is no unit conversion when using these constants - they are used as is
in the code, implicitly assuming the units are SI and untouched.
'''
units = {'gas_constant': 'J/(K mol)',
         'pressure': 'Pa',
         'min emul drop diameter': 'm',
         'max emul drop diameter': 'm',
         'acceleration': 'm/s^2'}
constants = {'gas_constant': 8.314,
             'atmos_pressure': 101325.0,
             'drop_min': 1.0e-6,
             'drop_max': 1.0e-5,
             #'g': 9.80665, # why not 6 sig figs?
             'gravity': 9.80665, # I like a longer name ;-)
             }

