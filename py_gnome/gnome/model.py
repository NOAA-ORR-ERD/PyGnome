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
    
    """ 
    PyGNOME Model Class
    
    Functionality:
        - Wind Movers (in the process of adding variable wind)
        - Diffusion Mover
        - CATS Mover (shio tides or constant flow, scaled to ref pos.)
        - Continuous Spill
        - Refloating
    """
    lw_bmp_dimensions = (1000, 1000)

    def __init__(self):
        """ 
            Initializes model attributes. Cannot yet call on the Cython module 
            to initialize the cpp model object.
        """
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
        
    def set_uncertain(self):
        c_gnome.set_model_uncertain()
        
    def add_map(self, image_size, bna_filename, refloat_halflife):
        """ 
            Adds both a color bitmap for visualization and a land-water map, for movement regulatoin and other tasks. 
        ++arguments:
            refloat_halflife is in seconds.
        """
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
            AH: I don't know that we'll need this?
        """
        i = self.movers.index(old_mover)
        self.movers[i] = new_mover
        return new_mover 
    
    def add_wind_mover(self, constant_wind_value):
        """ 
        Adds a constant wind. 
        
        :param constant_wind_value: (u, v) wind speed in  units of m/sec
        """
        m = c_gnome.wind_mover(constant_wind_value)
        self.movers.append(m)
        return m
        
    def add_random_mover(self, diffusion_coefficient):
        """
        adds a simple diffusion mover
        
        :param diffusion_coefficient:  units of cm^2/sec
        """
        
        self.movers.append(c_gnome.random_mover(diffusion_coefficient))
    
    def add_cats_mover(self, path, scale_type, shio_path_or_ref_position, scale_value=1.0, diffusion_coefficient=1.0):
        """ 
        Adds a GNOME CATS Mover to the model's list of movers.
        
        :param path: path to the CATS current pattern file (``*.CUR``)
        :param scale_type: ????
        :param shio_path_or_ref_position: shio_path_or_ref_position determines whether we're importing shio tides to be tied into the mover, or
            constantly scaling the river flow to a given reference position.
        :param scale_value=1.0: the scale vlaue used if there is not a SHIO file
        :param diffusion_coefficient=1.0: diffusion coefficent used for uncertaintly ?????


        :returns: `None`
        """
        mover = c_gnome.cats_mover(scale_type, scale_value, diffusion_coefficient)
        if not mover.read_topology(path):
            raise IOError("file: %s not found"%path)
        if(type(shio_path_or_ref_position)!=type("")):
            mover.set_ref_point(shio_path_or_ref_position)
            mover.set_velocity_scale(scale_value)
        else:
            if not mover.set_shio(shio_path_or_ref_position):
                raise IOError("file: %s not found"%shio_path_or_ref_position)
            mover.compute_velocity_scale()
        self.movers.append(mover)
        
    def set_run_duration(self, start_time, stop_time):
        """
        Sets the model start time, stop time and run duration using
        date-time strings with format: 'mm/dd/yyyy hh:mm:ss' 
        (Greenwich Mean Time)
        """
        # fixme: time zone?
        # fixme: use python datetimes???
        
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
        """
        Sets the model time step in seconds.
        """
        
        if self.duration == None:
            return
        self.interval_seconds = interval_seconds
        self.num_timesteps = floor(self.duration / self.interval_seconds)
        c_gnome.set_model_timestep(interval_seconds)

    def set_spill(self, num_particles, windage, (start_time, stop_time), (start_position, stop_position), \
                    disp_status=disp_status_dont_disperse, uncertain=False):
        # fixme: have windage range?
        """ 
            Sets a spill location.
        :params num_particles: number of particles to use
        :params windage: wind-drift factor of particles: fraction i.e. 0.03 for 3%
        :params (start_time, stop_time): (start_time, stop_time) start and stop time of release
        :params (start_position, stop_position): (start_position, stop_position) start and stop position of release
        :params disp_status: should the oil be dispersed?
        
        """
        allowable_spill = self.lw_map.allowable_spill_position
        if not (allowable_spill(start_position) and allowable_spill(stop_position)):
            print 'spill ignored: (' + str(start_position) + ', ' + str(stop_position) + ').'
            return
        try:
            self.spills += [spill.spill(self.lw_map, num_particles, disp_status, windage, \
                                            (greenwich.gwtm(start_time).time_seconds, greenwich.gwtm(stop_time).time_seconds), (start_position, stop_position), uncertain)]
        except:
            print 'Please check the format of your date time strings.'
            exit(-1)
        self.lwp_arrays += [numpy.copy(self.spills[len(self.spills)-1].npra['p'])]
    
    def reset(self):
        """ Resets model attributes """
        self.__init__()

    def release_particles(self):
        """ Releases particles depending on current model time. """
        model_time = self.start_time + self.time_step*self.interval_seconds
        for spill in self.spills:
            spill.release_particles(model_time)
            
    def refloat_particles(self):
        """ Refloats particles depending on lw_map's configured half-life."""
        spills = self.spills
        lwp_arrays = self.lwp_arrays
        for i in xrange(0, len(spills)):
            spills[i].refloat_particles(self.interval_seconds, lwp_arrays[i])


    def move_particles(self):
        """ 
        Moves particles, checks that they've not been landed, 
        and beaches them if they have.

        """
        lwpras = [] # last water position array
        spills = self.spills
        beach_element = self.lw_map.beach_element
        for spill in spills:
            lwpras += [numpy.copy(spill.npra['p'])]
        for mover in self.movers:
            for j in xrange(0, len(spills)):
                mover.get_move(self.interval_seconds, spills[j].npra, spills[j].uncertain, j+1) #1-indexed sets list
        for j in xrange(0, len(spills)):
            spill = spills[j]
            chromgph = spill.movement_check()
            for i in xrange(0, len(chromgph)):
                if chromgph[i]:
                    self.lwp_arrays[j][i] = lwpras[j][i]
                    beach_element(spill.npra['p'][i], lwpras[j][i])
   
    def initialize(self):
        """ Calls on Cython module to intialize the cpp model object. """
        c_gnome.initialize_model(self.spills)
        
    def step(self, output_dir="."):
        """ Steps the model forward in time. Needs support for hindcasting. """
        "step called: time step:", self.time_step
        if(self.duration == None):
            return False
        if(self.interval_seconds == None):
            return False
        if not len(self.movers) > 0:
            return False
        if self.time_step >= self.num_timesteps:
            return False
        c_gnome.step_model()
        self.time_step += 1
        self.release_particles()
        #self.refloat_particles()
        self.move_particles()
        filename = os.path.join(output_dir, 'map%05i.png'%self.time_step)
        print "filename:", filename
        self.c_map.draw_particles(self.spills, filename)
        return filename
        
