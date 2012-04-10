#!/usr/bin/env python

"""
script to test time aspects of GNOME with long island sound data

some timings:

start:

time_run: 2.45
time_init: 0.12
time_currents: 0.05
time_wind: 0.0
time_diffusion: 0.0
time_map: 0.38
time_draw: 2.02


with new map code:
( still not vectorized at all --  tiny bit faster, but not really )
time_run: 2.41
time_init: 0.13
time_currents: 0.06
time_wind: 0.0
time_diffusion: 0.0
time_map: 0.32
time_draw: 2.03

with to_pixel_array vectorized:

time_run: 2.56
time_init: 0.12
time_currents: 0.04
time_wind: 0.0
time_diffusion: 0.0
time_map: 0.28
time_draw: 2.24

with on_land_pixel_array vectorized

time_run: 1.81
time_init: 0.11
time_currents: 0.05
time_wind: 0.0
time_diffusion: 0.0
time_map: 0.01
time_draw: 1.75


"""

import os
import time
import numpy
from gnome import model
from gnome import c_gnome

#various parts:

time_run = 0.0
time_init = 0.0
time_currents = 0.0
time_wind = 0.0
time_diffusion = 0.0
time_map = 0.0
time_draw = 0.0

dimensions_bmp = (800, 500)
model_has_uncertainty = True

print "initializing the model"
start = time.clock()
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
mini_gnome.set_spill(spill['num_particles'],
                     spill['windage'],
                     (spill['start_time'], spill['stop_time']),
                     (spill['start_position'], spill['stop_position'])
                     )

if model_has_uncertainty:
    print "setting and uncertainty spill:"
    mini_gnome.set_spill(spill['num_particles']/10, spill['windage'], (spill['start_time'], spill['stop_time']), (spill['start_position'], spill['stop_position']), uncertain=True)

scale_type = 1
shio_file = "./SampleData/CLISShio.txt"
topology_file = "./SampleData/tidesWAC.CUR"


# fixme -- this should be a python datetime object!
model_start_time = '12/11/2012 06:00:00'
model_stop_time =  '12/12/2012 06:00:00'

mini_gnome.set_run_duration(model_start_time, model_stop_time)
mini_gnome.set_timestep(3600 * 2) # seconds

print "adding a wind mover:"
mini_gnome.add_wind_mover( (-1.0, -2.0) )

print "adding a random mover:"
mini_gnome.add_random_mover(10000)

print "adding a cats_mover:"
mini_gnome.add_cats_mover(topology_file, scale_type, shio_file, 1) # value needs to be changed here.

# create an images dir:
output_dir = "./images"
try:
    os.mkdir(output_dir)
except OSError:
    pass


print "running model:"

time_init += time.clock() - start

# run the model (doing it step by step for the timing)

start_run = time.clock()
while True:
    mini_gnome.initialize_model(None)
    "step called: time step:", mini_gnome.time_step
    if mini_gnome.time_step >= mini_gnome.num_timesteps:
        break
    mini_gnome.release_particles()

    #mini_gnome.refloat_particles()
    
    start = time.clock()
    lwpras = mini_gnome.move_particles()
    time_currents += time.clock() - start


    start = time.clock()
    mini_gnome.beach_particles(lwpras)
    time_map += time.clock() - start

    
    filename = os.path.join(output_dir, 'map%05i.png'%mini_gnome.time_step)
    print "filename:", filename

    start = time.clock()
    mini_gnome.c_map.draw_particles(mini_gnome.spills, filename)
    time_draw += time.clock() - start
    
    mini_gnome.time_step += 1
    result = c_gnome.step_model()
time_run = time.clock() - start_run


print "time_run:",  time_run
print "time_init:",  time_init
print "time_currents:",  time_currents
print "time_wind:",  time_wind
print "time_diffusion:",  time_diffusion
print "time_map:",  time_map
print "time_draw:",  time_draw
