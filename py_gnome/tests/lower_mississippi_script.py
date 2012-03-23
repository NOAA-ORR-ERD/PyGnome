#!/usr/bin/env python

import os
from gnome import model

dimensions_bmp = (800, 500)

spill = {'num_particles': 1000,
         'windage': .2,
         'start_time': '12/11/2012 06:55:00',
         'stop_time': '12/11/2012 06:55:00',
         'start_position': (-89.699944, 29.494558),
         'stop_position': (-89.699944, 29.494558),
         }

mini_gnome = model.Model()
mini_gnome.add_map(dimensions_bmp, "LMiss.bna", 5)
mini_gnome.set_spill(spill['num_particles'],
                     spill['windage'],
                     (spill['start_time'], spill['stop_time']),
                     (spill['start_position'], spill['stop_position']),
                     )

spill = {'num_particles': 1000,
         'windage': .02,
         'start_time': '12/11/2012 06:55:00',
         'stop_time': '12/11/2012 06:55:00',
         'start_position': (-89.699944, 29.494558),
         'stop_position': (-89.699944, 29.494558),
         }

cats_scale_type = 1
shio_file = "./CLISShio.txt"
cats_topology_file = "./LMiss.CUR"

model_start_time = '12/11/2012 06:55:00'
model_stop_time = '12/12/2012 06:59:00'

mini_gnome.set_run_duration(model_start_time, model_stop_time)
mini_gnome.set_timestep(900)
mini_gnome.set_spill(spill['num_particles'], spill['windage'], (spill['start_time'], spill['stop_time']), (spill['start_position'], spill['stop_position']))

#mini_gnome.add_wind_mover((-10, 15))
mini_gnome.add_random_mover(10000)
mini_gnome.add_cats_mover(cats_topology_file, cats_scale_type, (-89.699944, 29.494558), 1) # value needs to be changed here.

# create an images dir:
try:
    os.mkdir("./images")
except OSError:
    pass

while mini_gnome.step('./images') != False:
	pass
