#!/usr/bin/env python
"""
Script to show how to run py_gnome with weathering

This is a very simple script with "just weathering"

It has no land, no currents, and no transport.
"""

import os
from pathlib import Path
from gnome import scripting as gs


# define base directory -- so we can find the data files
base_dir = Path(__file__).parent
example_files = base_dir / 'example_files'
save_dir = base_dir / 'output'

print('initializing the model')
model = gs.Model(duration=gs.days(5))


print('adding outputters')
# This will write the total oil budget to a CSV file
model.outputters += gs.OilBudgetOutput(save_dir / 'GNOME_oil_budget.csv')

# This writes the detailed output to a netcdf file
# that is, all the properties of the elements
model.outputters += gs.NetCDFOutput(filename=save_dir / 'weathering_run.nc',
                                    which_data='standard',
                                    surface_conc=None
                                    )


print('adding a spill')
# We need a spill at the very least
oil_file = example_files / 'alaska-north-slope_AD00020.json'
spill = gs.surface_point_line_spill(num_elements=10,  # no need for a lot of elements for a instantaneous release
                                    start_position=(0.0, 0.0),  # position isn't important for this.
                                    release_time=model.start_time,
                                    amount=1000,
                                    units='bbl',
                                    substance=gs.GnomeOil(filename=oil_file),
                                    )

model.spills += spill

# print('adding a RandomMover:')
# model.movers += gs.RandomMover()

# print('adding a wind mover:')

# model.movers += gs.constant_point_wind_mover(speed=10, direction=0, units="m/s")

wind = gs.constant_wind(speed=10,
                        direction=0,
                        units='knots')
model.environment += wind

# Water properties are needed for the weathering algorithms
model.environment += gs.Water(25, units={"temperature": "C"})

# Waves are needed for dispersion -- it will use the wind defined above.
waves = gs.Waves()
model.environment += waves

print('adding the standard weatherers')
model.add_weathering()

print("running the model")
model.full_run()

# Saving the model as a "GNOME Save file"
model.save(saveloc=save_dir / 'WeatheringRun.gnome')


