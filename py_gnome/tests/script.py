#!/usr/bin/env python

from pyGNOME import model

dimensions_bmp = (1000,1000)
spills = [ [(19249, 41), (-73., 41.)], [1, 1500,], [2.0, 0.00,] ]

mini_gnome = model.Model()
mini_gnome.add_map(dimensions_bmp, "../utilities/LongIslandSoundMap.bna")
mini_gnome.set_spills(*spills)
mini_gnome.add_wind_mover((50000.00, 70000.00))
mini_gnome.add_random_mover(15.00)
mini_gnome.set_run_duration(0,500)
mini_gnome.set_timestep(10)
mini_gnome.create_environment()
while mini_gnome.step() != False:
	for spill in mini_gnome.live_particles:
		print spill[0]['p']
