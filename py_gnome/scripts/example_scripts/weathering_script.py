#!/usr/bin/env python
"""
Script to show how to run py_gnome with weathering

This is a "just weathering" run -- no land, currents, etc.
"""

import os
from pathlib import Path
from gnome import scripting as gs


# define base directory
base_dir = Path(__file__).parent
example_files = base_dir / 'example_files'
save_dir = base_dir / 'output'

def make_model():

    print('initializing the model')
    model = gs.Model(duration=gs.days(5))


    print('adding outputters')
    budget_file = save_dir / 'GNOME_oil_budget.csv'
    model.outputters += gs.OilBudgetOutput(budget_file)

    print('adding a spill')
    # We need a spill at the very least
    oil_file = example_files / 'alaska-north-slope_AD00020.json'
    spill = gs.surface_point_line_spill(num_elements=10,  # no need for a lot of elements for a instantaneous release
                                        start_position=(0.0, 0.0, 0.0),
                                        release_time=model.start_time,
                                        amount=1000,
                                        substance=gs.GnomeOil(filename=oil_file),
                                        units='bbl')

    model.spills += spill

    print('adding a RandomMover:')
    model.movers += gs.RandomMover()

    print('adding a wind mover:')

    model.movers += gs.constant_wind_mover(speed=10, direction=0, units="m/s")


    model.environment += gs.Water(25, units={"temperature": "C"})

    waves = gs.Waves()
    model.environment += waves

    print('adding the standard weatherers')
    model.add_weathering()

    return model


if __name__ == "__main__":
    model = make_model()
    print("running the model")
    model.full_run()
    model.save(saveloc=save_dir / 'WeatheringRun.gnome')


