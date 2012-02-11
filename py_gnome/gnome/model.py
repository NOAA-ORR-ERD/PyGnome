 
import numpy
import random
from map import gnome_map
import os
import sys
from math import floor
import collections
from gnome import c_gnome
from basic_types import *
import spill

    
class Model:
    
    """ Documentation goes here. """

    def __init__(self):
        self.movers = collections.deque()
        self.gnome_map = None
        self.particles = collections.deque()
        self.live_particles = collections.deque()
        self.start_time = None
        self.stop_time = None
        self.duration = None
        self.interval_seconds = None
        self.num_timesteps = None
        self.time_step = 0
        self.spills = []
        self.lwp_arrays = []
        
    def add_map(self, image_size, bna_filename, refloat_halflife):
        self.gnome_map = gnome_map(image_size, bna_filename, refloat_halflife)
    
    def add_wind_mover(self, constant_wind_value):
        self.movers.append(c_gnome.wind_mover(constant_wind_value))
        
    def add_random_mover(self, diffusion_coefficient):
        self.movers.append(c_gnome.random_mover(diffusion_coefficient))
        
    def set_run_duration(self, start_time, stop_time):
        if not start_time < stop_time:
            return
        self.start_time = start_time
        self.stop_time = stop_time
        self.duration = stop_time - start_time
    
    def set_timestep(self, interval_seconds):
        if self.duration == None:
            return
        self.interval_seconds = interval_seconds
        self.num_timesteps = floor(self.duration / self.interval_seconds)

    def set_spill(self, num_particles, windage, (start_time, stop_time), (start_position, stop_position), \
                    disp_status=disp_status_dont_disperse):
        allowable_spill = self.gnome_map.allowable_spill_position
        if not (allowable_spill(start_position) and allowable_spill(stop_position)):
            print 'spill ignored: (' + str(start_position) + ', ' + str(stop_position) + ').'
            return
        self.spills += [spill.spill(self.gnome_map, num_particles, disp_status, windage, \
                                        (start_time, stop_time), (start_position, stop_position))]
        self.lwp_arrays += [numpy.copy(self.spills[len(self.spills)-1].npra['p'])]
    
    def reset_steps(self):
        self.time_step = 0

    def release_particles(self):
        model_time = self.start_time + self.time_step*self.interval_seconds
        for spill in self.spills:
            spill.release_particles(model_time)
            
    def refloat_particles(self):
        spills = self.spills
        lwp_arrays = self.lwp_arrays
        for i in xrange(0, len(spills)):
            spills[i].refloat_particles(self.interval_seconds, lwp_arrays[i])
    
    def beach_element(self, p, lwp):
        in_water = self.gnome_map.in_water
        while not in_water((p['p_long'], p['p_lat'])):
            displacement = (p['p_long'] - lwp['p_long'], p['p_lat'] - lwp['p_lat'])
            displacement /= 2
            p['p_long'] = lwp['p_long'] + displacement[0]
            p['p_lat'] = lwp['p_lat'] + displacement[1]
        
    def move_particles(self):
        def set_self_chromgph(obj,chromgph):
            obj.chromgph = chromgph
            set_self_chromgph = do_nothing
        def do_nothing(null,nil): 
            pass
        lwpras = []
        spills = self.spills
        for spill in spills:
            lwpras += [numpy.copy(spill.npra['p'])]
        for mover in self.movers:
            for j in xrange(0, len(spills)):
                mover.get_move(self.interval_seconds, spills[j].npra)
        for j in xrange(0, len(spills)):
            spill = spills[j]
            chromgph = spill.movement_check(set_self_chromgph)
            for i in xrange(0, len(chromgph)):
                if chromgph[i]:
                    self.lwp_arrays[j][i] = lwpras[j][i]
                    self.beach_element(spill.npra['p'][i], lwpras[j][i])
                    
    def step(self):
        if(self.duration == None):
            return False
        if(self.interval_seconds == None):
            return False
        if not len(self.movers) > 0:
            return False
        if self.time_step >= self.num_timesteps:
            return False
        self.release_particles()
        self.refloat_particles()
        self.move_particles()
        self.gnome_map.draw_particles(self.spills, '_map'+str(self.time_step)+'.png')
        self.time_step += 1
        
