#!/usr/bin/python

"""

NOTE: TEST IS DEPRECATED -- needs re-doing for new code structure

"""


import numpy as np
from math import sin,cos,pi
from random import random
from gnome.cy_gnome import cy_cats_mover
from gnome import basic_types
from gnome import greenwich

scale_type = 1
shio_file = "../scripts/SampleData/CLISShio.txt"
topology_file = "../scripts/SampleData/LI_tidesWAC.CUR"

cats_mover = cy_cats_mover.Cy_cats_mover(scale_type)
cats_mover.read_topology(topology_file)
cats_mover.set_shio(shio_file)


model_time = greenwich.gwtm('01/01/1970 11:00:00').time_seconds

#################
# create arrays #
#################

wp_ra = np.empty((10,), dtype=basic_types.world_point)
ref_ra = np.empty((10,), dtype=basic_types.world_point)
uncertain_ra = np.empty((10,), dtype=basic_types.le_uncertain_rec)	# one uncertain rec per le


################
# init. arrays #
################

N = len(wp_ra)

wp_ra[:]['long'] = (-72.419992)
wp_ra[:]['lat'] = (41.202120)
wp_ra[:]['z'] = 0.
ref_ra[:]['long'] = (-72.419992)
ref_ra[:]['lat'] = (41.202120)
ref_ra[:]['z'] = 0.

ref_ra[:]['long'] *= 1000000
ref_ra[:]['lat'] *= 1000000

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
    cats_mover.get_move(N, model_time, 10, ref_ra, wp_ra)


print wp_ra
print

print '#################'
print '# uncertainmove #'
print '#################'

wp_ra[:]['long'] = (-72.419992)
wp_ra[:]['lat'] = (41.202120)
wp_ra[:]['z'] = 0.
ref_ra[:]['long'] = (-72.419992)
ref_ra[:]['lat'] = (41.202120)
ref_ra[:]['z'] = 0.

ref_ra[:]['lat'] *= 1000000
ref_ra[:]['long'] *= 1000000

for x in range(0, 10):
    cats_mover.get_move_uncertain(N, model_time, 10, ref_ra, wp_ra, uncertain_ra)

print wp_ra
