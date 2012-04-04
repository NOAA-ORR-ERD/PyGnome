#!/usr/bin/env python

from gnome import model

dimensions_bmp = (1200,1200)

spill = {'num_particles': 2,
         'windage': .2,
         'start_time': 0,
         'stop_time': 0,
         'start_position': (-72.719832,41.112120),
         'stop_position': (-72.719832,41.222120),
         }

mini_gnome = model.Model()
mini_gnome.add_map(dimensions_bmp, "LongIslandSoundMap.bna", 300)
mini_gnome.set_spill(spill['num_particles'], spill['windage'], (spill['start_time'], spill['stop_time']), (spill['start_position'], spill['stop_position']))

mini_gnome.add_wind_mover((0, 1)) #U and V components.
mini_gnome.add_random_mover(10000000) #Diffusion coeff. in cm^2/s
TimeStep=150*60
mini_gnome.set_run_duration(0,3*TimeStep) #Start time = 0 seconds from epoc.  Stop time = 86400 seconds from epoc.
mini_gnome.set_timestep(TimeStep) #Time-step = 10 seconds.


while mini_gnome.step() != False:
    print mini_gnome.spills[0].npra["p"]
    pass