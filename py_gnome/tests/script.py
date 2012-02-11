#!/usr/bin/env python

from gnome import model

dimensions_bmp = (1200,1200)
spill = {'num_particles': 1000,
         'windage': .2,
         'start_time': 0,
         'stop_time': 0,
         'start_position': (-72.719832,41.112120),
         'stop_position': (-72.719832,41.222120),
         }

mini_gnome = model.Model()
mini_gnome.add_map(dimensions_bmp, "LongIslandSoundMap.bna", 300)
mini_gnome.set_spill(spill['num_particles'], spill['windage'], (spill['start_time'], spill['stop_time']), (spill['start_position'], spill['stop_position']))

spill = {'num_particles': 1000,
         'windage': .2,
         'start_time': 0,
         'stop_time': 0,
         'start_position': (-72.509832,41.112120),
         'stop_position': (-72.409832,41.112120),
         }

mini_gnome.set_spill(spill['num_particles'], spill['windage'], (spill['start_time'], spill['stop_time']), (spill['start_position'], spill['stop_position']))
mini_gnome.add_wind_mover((-100, 150))
mini_gnome.add_random_mover(10000000)
mini_gnome.set_run_duration(0,1000)
mini_gnome.set_timestep(10)

while mini_gnome.step() != False:
	pass
