
from gnome.array_types import gat

'''
gnome.ops is a collection of functionized model operations. They are all static functions
that can be imported and run anywhere, as long as the execution context can satisfy the function
parameter requirements.

aggregated_data contains functions for computing model metadata in aggregate.
density contains functions for initializing and recalculating LE density
viscosity contains functions for initializing and recalculating LE viscosity
'''

weathering_array_types = {'fate_status': gat('fate_status'),
                    'positions': gat('positions'),
                    'status_codes': gat('status_codes'),
                    'density': gat('density'),
                    'viscosity': gat('viscosity'),
                    'mass_components': gat('mass_components'),
                    'mass': gat('mass'),
                    'oil_density': gat('oil_density'),
                    'oil_viscosity': gat('oil_viscosity'),
                    'init_mass': gat('init_mass'),
                    'frac_water': gat('frac_water'),
                    'frac_lost': gat('frac_lost'),	# change to frac_dissolved
                    'frac_evap': gat('frac_evap'),
                    'age': gat('age')}

non_weathering_array_types = {'fate_status': gat('fate_status'),
                    'positions': gat('positions'),
                    'status_codes': gat('status_codes'),
                    'density': gat('density'),
                    'mass': gat('mass'),
                    'init_mass': gat('init_mass'),
                    'age': gat('age')}