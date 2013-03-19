import json
from datetime import datetime,timedelta
import pprint

import numpy as np
import colander

import gnome
from gnome import movers
from gnome.persist import scenario

"""
Define a scenario and persist it to ./test_persist/
"""
saveloc = './test_persist'
start_time = datetime(2013, 2, 13, 9, 0)
model = gnome.model.Model(start_time = start_time,
                        duration = timedelta(days=2),
                        time_step = 30 * 60, # 1/2 hr in seconds
                        uncertain = False,
                        )

print "adding a spill"

model.spills += gnome.spill.SurfaceReleaseSpill(num_elements=1000,
                                        start_position = (144.664166, 13.441944, 0.0),
                                        release_time = start_time,
                                        end_release_time = start_time + timedelta(hours=6)
                                        )

#need a scenario for SimpleMover
model.movers += movers.simple_mover.SimpleMover(velocity=(1.0, -1.0, 0.0))
model.movers += gnome.movers.RandomMover(diffusion_coef=100000)

series = np.array( (start_time, ( 10,   45) ),  dtype=gnome.basic_types.datetime_value_2d).reshape((1,))
model.environment += gnome.environment.Wind(timeseries=series, units='meter per second')

model.movers += gnome.movers.WindMover( [w for w in model.environment][0] )

print "saving .."
scenario.save(model,saveloc)
print "loading .."
#model2 = scenario.load(saveloc,'model_{0}.txt'.format(model.id))
