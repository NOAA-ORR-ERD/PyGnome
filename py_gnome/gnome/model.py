import numpy
import random
import os
import sys
from math import floor
import collections
from gnome import c_gnome, greenwich
from map import gnome_map, lw_map
from basic_types import disp_status_dont_disperse
import spill

    
class Model:
    
    """ Documentation goes here. """
    
    lw_bmp_dimensions = (1000, 1000)

    def __init__(self):
        self.movers = []
        self.c_map = None
        self.lw_map = None
        self.start_time = None
        self.stop_time = None
        self.duration = None
        self.interval_seconds = None
        self.num_timesteps = None
        self.time_step = 0
        self.spills = []
        self.lwp_arrays = []
        self.shio = None
        
    def add_map(self, image_size, bna_filename, refloat_halflife):
        if not refloat_halflife:
            print 'Refloat halflife must be nonzero.'
            exit(-1)
        self.c_map = gnome_map(image_size, bna_filename)
        self.lw_map = lw_map(self.lw_bmp_dimensions, bna_filename, refloat_halflife, '1')
    
    def add_mover(self, mover):
        """
        add a new mover to the model -- at the end of the stack
        """
        self.movers.append(mover)

    def remove_mover(self, mover):
        """
        remove the passed-in mover from the mover list
        """
        self.movers.remove(mover)

    def replace_mover(self, old_mover, new_mover):
        """
        replace a given mover with a new one
        """
        i = self.movers.index(old_mover)
        self.movers[i] = new_mover
        return new_mover 
    
    def add_wind_mover(self, constant_wind_value):
        m = c_gnome.wind_mover(constant_wind_value)
        self.movers.append(m)
        return m
        
    def add_random_mover(self, diffusion_coefficient):
        """
        adds a simple diffusion mover
        
        diffusion_coefficient in units of cm^2/sec
        """
        self.movers.append(c_gnome.random_mover(diffusion_coefficient))
    
    def add_cats_mover(self, path, scale_type, shio_path_or_ref_position, scale_value=1, diffusion_coefficient=1):
        mover = c_gnome.cats_mover(scale_type, scale_value, diffusion_coefficient, shio_path_or_ref_position)
        mover.read_topology(path)
        if(type(shio_path_or_ref_position)!=type("")):
            mover.set_ref_point(shio_path_or_ref_position)
            mover.set_velocity_scale(scale_value)
        else:
            mover.compute_velocity_scale()
        self.movers.append(mover)
        
    def set_run_duration(self, start_time, stop_time):
    
        """
        Now using date-time strings, 'yyyy-mm-dd hh:mm:ss' (Greenwich Mean Time)
        
        """
        
        try:
            start_time = greenwich.gwtm(start_time).time_seconds
            stop_time = greenwich.gwtm(stop_time).time_seconds
        except:
            print 'Please check the format of your date time strings.'
            exit(-1)
            
        if not start_time < stop_time:
            print 'Please check your start and stop times.'
            exit(-1)
            
        self.start_time = start_time
        self.stop_time = stop_time
        self.duration = stop_time - start_time
        c_gnome.set_model_start_time(start_time)
        c_gnome.set_model_time(start_time)
        c_gnome.set_model_duration(self.duration)
    
    def set_timestep(self, interval_seconds):
        if self.duration == None:
            return
        self.interval_seconds = interval_seconds
        self.num_timesteps = floor(self.duration / self.interval_seconds)
        c_gnome.set_model_timestep(interval_seconds)

    def set_spill(self, num_particles, windage, (start_time, stop_time), (start_position, stop_position), \
                    disp_status=disp_status_dont_disperse):
        allowable_spill = self.lw_map.allowable_spill_position
        if not (allowable_spill(start_position) and allowable_spill(stop_position)):
            print 'spill ignored: (' + str(start_position) + ', ' + str(stop_position) + ').'
            return
        try:
            self.spills += [spill.spill(self.lw_map, num_particles, disp_status, windage, \
                                            (greenwich.gwtm(start_time).time_seconds, greenwich.gwtm(stop_time).time_seconds), (start_position, stop_position))]
        except:
            print 'Please check the format of your date time strings.'
            exit(-1)
        self.lwp_arrays += [numpy.copy(self.spills[len(self.spills)-1].npra['p'])]
    
    def reset(self):
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
        in_water = self.lw_map.in_water
        displacement = ((p['p_long'] - lwp['p_long']), (p['p_lat'] - lwp['p_lat']))
        while not in_water((p['p_long'], p['p_lat'])):
            displacement = (displacement[0]/2, displacement[1]/2)
            p['p_long'] = lwp['p_long'] + displacement[0]
            p['p_lat'] = lwp['p_lat'] + displacement[1]

    def move_particles(self):
        lwpras = []
        spills = self.spills
        for spill in spills:
            lwpras += [numpy.copy(spill.npra['p'])]
        for mover in self.movers:
            for j in xrange(0, len(spills)):
                mover.get_move(self.interval_seconds, spills[j].npra)
        for j in xrange(0, len(spills)):
            spill = spills[j]
            chromgph = spill.movement_check()
            for i in xrange(0, len(chromgph)):
                if chromgph[i]:
                    self.lwp_arrays[j][i] = lwpras[j][i]
                    self.beach_element(spill.npra['p'][i], lwpras[j][i])
                    
    def step(self, output_dir="."):
        "step called: time step:", self.time_step
        if(self.duration == None):
            return False
        if(self.interval_seconds == None):
            return False
        if not len(self.movers) > 0:
            return False
        if self.time_step >= self.num_timesteps:
            return False
        self.release_particles()
        #self.refloat_particles()
        self.move_particles()
        filename = os.path.join(output_dir, 'map%05i.png'%self.time_step)
        print "filename:", filename
        self.c_map.draw_particles(self.spills, filename)
        self.time_step += 1
        c_gnome.step_model()
        return filename
        
