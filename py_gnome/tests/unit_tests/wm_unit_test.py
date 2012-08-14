#!/usr/bin/env python
import numpy as np
from math import sin,cos,pi
from random import random


from gnome import basic_types
from gnome import wind_mover
from gnome import greenwich
wm = wind_mover.wind_mover()

start_time = greenwich.gwtm('01/01/1970 10:00:00').time_seconds
stop_time = greenwich.gwtm('01/01/1970 12:00:00').time_seconds
model_time = greenwich.gwtm('01/01/1970 11:00:00').time_seconds


#################
# create arrays #
#################

wp_ra = np.empty((10,), dtype=basic_types.world_point_3d)
ref_ra = np.empty((10,), dtype=basic_types.world_point_3d)
wind_ra = np.empty((10,), dtype=np.double)
disp_ra = np.empty((10,), dtype=np.short)
time_vals = np.empty((1,), dtype=basic_types.time_value_pair)
uncertain_ra = np.empty((10,), dtype=basic_types.wind_uncertain_rec)	# one uncertain rec per le

f_sigma_theta = 1  # ?? 
f_sigma_vel = 1     # ??

################
# init. arrays #
################

N = len(wp_ra)
M = len(time_vals)

wp_ra[:] = 1.
ref_ra[:] = 1.

ref_ra[:]['p']['p_lat'] *= 1000000
ref_ra[:]['p']['p_long'] *= 1000000

wind_ra[:] = 1
disp_ra[:] = 0
time_vals['value']['u'] = 10000
time_vals['value']['v'] = 0

   # initialize uncerainty array:

for x in range(0, N):
    theta = random()*2*pi
    uncertain_ra[x]['randCos'] = cos(theta)
    uncertain_ra[x]['randSin'] = sin(theta)
    
    # ein uncertain array

##################
# call and check #
##################

breaking_wave = 10 #?? 
mix_layer_depth = 10 #??


print '###################'
print '# init. positions #'
print '###################'

print wp_ra
print


print '#################'
print '# forecast move #'
print '#################'

for x in range(0, 1):
    wm.get_move(N, model_time, 10, ref_ra, wp_ra, wind_ra, disp_ra, breaking_wave, mix_layer_depth, time_vals)

print wp_ra

print
print '#################'
print '# uncertainmove #'
print '#################'

wp_ra[:] = 1.
ref_ra[:] = 1.

ref_ra[:]['p']['p_lat'] *= 1000000
ref_ra[:]['p']['p_long'] *= 1000000

for x in range(0, 1):
    wm.get_move_uncertain(N, model_time, 10, ref_ra, wp_ra, wind_ra, disp_ra, f_sigma_vel, f_sigma_theta, breaking_wave, mix_layer_depth, uncertain_ra, time_vals)

print wp_ra
