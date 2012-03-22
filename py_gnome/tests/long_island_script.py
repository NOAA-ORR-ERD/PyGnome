#!/usr/bin/env python

import os
from gnome import model

dimensions_bmp = (800, 500)

mini_gnome = model.Model()
mini_gnome.add_map(dimensions_bmp, "LongIslandSoundMap.BNA", 5)

spill = {'num_particles': 1000,
         'windage': .2,
         'start_time': '2011-11-12 06:55:00',
         'stop_time': '2011-11-12 06:55:00',
         'start_position': (-72.419882,41.202120),
         'stop_position': (-72.419882,41.202120),
         }

mini_gnome.set_spill(spill['num_particles'],
                     spill['windage'],
                     (spill['start_time'], spill['stop_time']),
                     (spill['start_position'], spill['stop_position']),
                     )

spill = {'num_particles': 1000,
         'windage': .02,
         'start_time': '2011-11-12 06:55:00',
         'stop_time': '2011-11-12 06:55:00',
         'start_position': (-72.419882,41.202120),
         'stop_position': (-72.419882,41.202120),
         }

mini_gnome.set_spill(spill['num_particles'], spill['windage'], (spill['start_time'], spill['stop_time']), (spill['start_position'], spill['stop_position']))

scale_type = 1
shio_file = "./CLISShio.txt"
topology_file = "./tidesWAC.CUR"


model_start_time = '2011-11-12 06:55:00'
model_stop_time = '2011-12-12 06:59:00'

mini_gnome.set_run_duration(model_start_time, model_stop_time)
mini_gnome.set_timestep(900)

#mini_gnome.add_wind_mover((-10, 15))
mini_gnome.add_random_mover(10000)
mini_gnome.add_cats_mover(topology_file, scale_type, shio_file, 1) # value needs to be changed here.

# create an images dir:
try:
    os.mkdir("./images")
except OSError:
    pass

while mini_gnome.step('./images') != False:
	pass
