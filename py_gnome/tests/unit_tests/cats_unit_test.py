#!/usr/bin/python
import numpy as np
from math import sin,cos,pi
from random import random
from gnome import cats_mover
from gnome import basic_types
from gnome import greenwich

scale_type = 1
shio_file = "../scripts/SampleData/CLISShio.txt"
topology_file = "../scripts/SampleData/LI_tidesWAC.CUR"

cats_mover = cats_mover.cats_mover(scale_type)
cats_mover.read_topology(topology_file)
cats_mover.set_shio(shio_file)


start_time = greenwich.gwtm('01/01/1970 10:00:00').time_seconds
stop_time = greenwich.gwtm('01/01/1970 12:00:00').time_seconds
model_time = greenwich.gwtm('01/01/1970 11:00:00').time_seconds

#################
# create arrays #
#################

wp_ra = np.empty((10,), dtype=basic_types.world_point_3d)
ref_ra = np.empty((10,), dtype=basic_types.world_point_3d)
uncertain_ra = np.empty((10,), dtype=basic_types.le_uncertain_rec)	# one uncertain rec per le


################
# init. arrays #
################

N = len(wp_ra)

wp_ra[:]['p'] = (-72.419992,41.202120)
wp_ra[:]['z'] = 0.
ref_ra[:]['p'] = (-72.419992,41.202120)
ref_ra[:]['z'] = 0.

ref_ra[:]['p']['p_lat'] *= 1000000
ref_ra[:]['p']['p_long'] *= 1000000

   # initialize uncerainty array:

for x in range(0, N):
    theta = random()*2*pi
    uncertain_ra[x]['downStream'] = sin(theta)
    uncertain_ra[x]['crossStream'] = cos(theta)
    
    # ein uncertain array

##################
# call and check #
##################

print '###################'
print '# init. positions #'
print '###################'

print wp_ra
print

print '#################'
print '# forecast move #'
print '#################'

for x in range(0, 10):
    cats_mover.get_move(N, start_time, stop_time, model_time, 10, ref_ra, wp_ra)


print wp_ra
print

print '#################'
print '# uncertainmove #'
print '#################'

wp_ra[:]['p'] = (-72.419992,41.202120)
wp_ra[:]['z'] = 0.
ref_ra[:]['p'] = (-72.419992,41.202120)
ref_ra[:]['z'] = 0.

ref_ra[:]['p']['p_lat'] *= 1000000
ref_ra[:]['p']['p_long'] *= 1000000

for x in range(0, 10):
    cats_mover.get_move_uncertain(N, start_time, stop_time, model_time, 10, ref_ra, wp_ra, uncertain_ra)

print wp_ra
