#!/usr/bin/env python
"""
Script to show how to run py_gnome with weathering

This is a "just weathering" run -- no land, currents, etc.
"""

from datetime import datetime, timedelta

from gnome import scripting as gs

# define base directory
# base_dir = os.path.dirname(__file__)

# water = Water(280.928)
# wind = constant_wind(20., 117, 'knots')
# waves = Waves(wind, water)

def make_model():
    print 'initializing the model'

#    start_time = datetime(2015, 5, 14, 0, 0)

    model = gs.Model(duration=gs.days(5))


    print 'adding outputters'

    model.outputters += gs.OilBudgetOutput("")

    print 'adding a spill'
    # We need a spill at the very least

    spill = gs.point_line_release_spill(num_elements=10,  # no need for a lot of elements for a instantaneous release
                                        start_position=(0.0, 0.0, 0.0),
                                        release_time=model.start_time,
                                        amount=1000,
                                        substance='ALASKA NORTH SLOPE (MIDDLE PIPELINE, 1997)',
                                        units='bbl')

    model.spills += spill

    print 'adding a RandomMover:'
    model.movers += gs.RandomMover()

    print 'adding a wind mover:'

    model.movers += gs.constant_wind_mover(speed=10, direction=0, units="m/s")


    model.environment += gs.Water(25, units={"temperature": "C"})
    #waves = Waves(wind, water)
    waves = gs.Waves()
    model.environment += waves

    print 'adding the standard weatherers'

    model.add_weathering()

    return model


if __name__ == "__main__":
    model = make_model()
    model.full_run()
    model.save('.')
