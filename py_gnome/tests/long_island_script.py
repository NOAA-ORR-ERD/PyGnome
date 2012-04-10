#!/usr/bin/env python

"""
script to test GNOME with long island sound data
"""

import os
from gnome import model

dimensions_bmp = (800, 500)
model_has_uncertainty = True

print "initializing the model"

mini_gnome = model.Model()

print "adding the map"

mini_gnome.add_map(dimensions_bmp, "./SampleData/LongIslandSoundMap.BNA", 5)


spill = {'num_particles': 1000,
         'windage': .2,
         'start_time': '12/11/2012 06:55:00',
         'stop_time': '12/11/2012 06:55:00',
         'start_position': (-72.419992,41.202120),
         'stop_position': (-72.419992,41.202120),
         }

print "setting the first spill"

mini_gnome.set_spill(spill['num_particles'],
                     spill['windage'],
                     (spill['start_time'], spill['stop_time']),
                     (spill['start_position'], spill['stop_position']),
                     )



spill = {'num_particles': 1000,
         'windage': .2,
         'start_time': '12/11/2012 06:55:00',
         'stop_time': '12/11/2012 06:55:00',
         'start_position': (-72.419992,41.202120),
         'stop_position': (-72.419992,41.202120),
         }

print "setting the seconds spill:"
mini_gnome.set_spill(spill['num_particles'], spill['windage'], (spill['start_time'], spill['stop_time']), (spill['start_position'], spill['stop_position']))

if model_has_uncertainty:
    print "setting and uncertainty spill:"
    mini_gnome.set_uncertain()
    mini_gnome.set_spill(spill['num_particles']/10, spill['windage']/10000, (spill['start_time'], spill['stop_time']), (spill['start_position'], spill['stop_position']), uncertain=True)

scale_type = 1
shio_file = "./SampleData/CLISShio.txt"
topology_file = "./SampleData/tidesWAC.CUR"


model_start_time = '12/11/2012 06:55:00'
model_stop_time = '12/13/2012 06:59:00'

mini_gnome.set_run_duration(model_start_time, model_stop_time)
mini_gnome.set_timestep(900)
mini_gnome.initialize()
print "adding a wind mover:"
mini_gnome.add_wind_mover((-.1, -.05))

print "adding a random mover:"
mini_gnome.add_random_mover(10000)

print "adding a cats_mover:"
mini_gnome.add_cats_mover(topology_file, scale_type, shio_file, 1) # value needs to be changed here.

# create an images dir:
try:
    os.mkdir("./images")
except OSError:
    pass

print "running model:"

while mini_gnome.step('./images') != False:
	pass
