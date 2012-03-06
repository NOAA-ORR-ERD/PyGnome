#!/usr/bin/env python

import os
from gnome import model

dimensions_bmp = (800, 500)

spill = {'num_particles': 1000,
         'windage': .2,
         'start_time': '2011-11-12 09:55:00',
         'stop_time': '2011-11-12 09:55:00',
         'start_position': (-72.419882,41.152120),
         'stop_position': (-72.389882,41.172120),
         }

mini_gnome = model.Model()
mini_gnome.add_map(dimensions_bmp, "LongIslandSoundMap.bna", 300)
mini_gnome.set_spill(spill['num_particles'],
                     spill['windage'],
                     (spill['start_time'], spill['stop_time']),
                     (spill['start_position'], spill['stop_position']),
                     )

spill = {'num_particles': 1000,
         'windage': .02,
         'start_time': '2011-11-12 09:55:00',
         'stop_time': '2011-11-12 10:00:00',
         'start_position': (-72.509832,41.112120),
         'stop_position': (-72.409832,41.112120),
         }

cats_scale_type = 1
cats_ref_position = (-72.705, 41.2275)
shio_file = "./CLISShio.txt"
cats_topology_file = "./tidesWAC.CUR"

model_start_time = '2011-11-12 09:55:00'
model_stop_time = '2011-11-12 11:00:00'

mini_gnome.set_run_duration(model_start_time, model_stop_time)
mini_gnome.set_timestep(10)
mini_gnome.set_spill(spill['num_particles'], spill['windage'], (spill['start_time'], spill['stop_time']), (spill['start_position'], spill['stop_position']))

#mini_gnome.add_wind_mover((-100, 150))
mini_gnome.add_random_mover(10000000)
mini_gnome.add_cats_mover(cats_topology_file, cats_scale_type, cats_ref_position, shio_file, 10)

# create an images dir:
try:
    os.mkdir("./images")
except OSError:
    pass

while mini_gnome.step('./images') != False:
	pass
