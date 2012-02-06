#!/usr/bin/env python


dimensions_bmp = (1000,1000)
spills = [ [(19249, 41), (-73, 41)], [1, 1500.], [2.0, 0.00,] ]

from pyGNOME import model
mini_gnome = model.Model()
mini_gnome.add_map(dimensions_bmp, "../utilities/LongIslandSoundMap.bna")
mini_gnome.set_spills(*spills)
mini_gnome.add_wind_mover((5.00, 7.00))
mini_gnome.add_random_mover(2.00)
mini_gnome.set_run_duration(0,500)
mini_gnome.set_timestep(10)
mini_gnome.create_environment()
while mini_gnome.step() != False:
	print mini_gnome.time_step
