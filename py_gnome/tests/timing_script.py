#!/usr/bin/env python

"""
script to test time aspects of GNOME with long island sound data
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
mini_gnome.add_wind_mover((-.5, -.2))

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
    
    lwpras = [] # last water position array
    spills = mini_gnome.spills
    beach_element = mini_gnome.lw_map.beach_element
    for spill in spills:
        lwpras += [numpy.copy(spill.npra['p'])]

    start = time.clock()
    for mover in mini_gnome.movers:
        for j in xrange(0, len(spills)):
            mover.get_move(mini_gnome.interval_seconds, spills[j].npra, spills[j].uncertain)
    time_currents += time.clock() - start

    start = time.clock()
    for j in xrange(0, len(spills)):
        spill = spills[j]
        chromgph = spill.movement_check()
        for i in xrange(0, len(chromgph)):
            if chromgph[i]:
                mini_gnome.lwp_arrays[j][i] = lwpras[j][i]
                beach_element(spill.npra['p'][i], lwpras[j][i])
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
