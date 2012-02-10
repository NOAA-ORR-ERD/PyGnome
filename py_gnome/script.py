#!/usr/bin/env python

from gnome import model

dimensions_bmp = (1000,1000)
spill = {'num_particles': 1000,
     'windage': .2,
     'start_time': 0,
     'stop_time': 0,
     'start_position': (-73,41),
     'stop_position': (-73,41),
    }

mini_gnome = model.Model()
mini_gnome.add_map(dimensions_bmp, "LongIslandSoundMap.bna", 300)
mini_gnome.set_spill(spill['num_particles'], spill['windage'], (spill['start_time'], spill['stop_time']), (spill['start_position'], spill['stop_position']))
mini_gnome.add_wind_mover((500.00, 700.00))
mini_gnome.add_random_mover(50.)
mini_gnome.set_run_duration(0,500)
mini_gnome.set_timestep(10)

while mini_gnome.step() != False:
    for spill in mini_gnome.spills:
	print spill.npra['p']
