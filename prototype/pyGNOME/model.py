 
import numpy
import random                
import map
import os
import sys
from math import floor
sys.path[len(sys.path):] = [os.environ['HOME']+'/Workspace/GNOME/prototype']	# ...
from cyGNOME import c_gnome

world_point = numpy.dtype([('p_long', numpy.int), ('p_lat', numpy.int)], align=True)
world_point3d = numpy.dtype([('p', world_point), ('z', numpy.double)], align=True)
world_rect = numpy.dtype([('lo_long', numpy.long), ('lo_lat', numpy.long), \
                            ('hi_long', numpy.long), ('hi_lat', numpy.long)], align=True)
le_rec = numpy.dtype([('le_units', numpy.int), ('le_key', numpy.int), ('le_custom_data', numpy.int), \
            ('p', world_point), ('z', numpy.double), ('release_time', numpy.uint), \
            ('age_in_hrs_when_released', numpy.double), ('clock_ref', numpy.uint), \
            ('pollutant_type', numpy.short), ('mass', numpy.double), ('density', numpy.double), \
            ('windage', numpy.double), ('droplet_size', numpy.int), ('dispersion_status', numpy.short), \
            ('rise_velocity', numpy.double), ('status_code', numpy.short), ('last_water_pt', world_point), ('beach_time', numpy.uint)], align=True)

class Model:
    
    """ Documentation goes here. """

    def __init__(self):
        self.movers = []
        self.map = None
        self.particles = []
        self.live_particles = []
        self.start_time = None
        self.stop_time = None
        self.duration = None
        self.interval_seconds = None
        self.num_timesteps = None
        
    def add_mover(self, type):
    	pass
    	
    def add_map(self, image_size, bna_filename):
        self.map = map(image_size, bna_filename)
    
    def add_wind_mover(self, constant_wind_value):
        self.movers += [c_gnome.wind_mover(constant_wind_value)]
        
    def add_random_mover(self, diffusion_coefficient):
        self.movers += [c_gnome.random_mover(diffusion_coefficient)]
        
    def set_run_duration(self, start_time, stop_time):
    	if not start_time < stop_time:
    		return
    	self.start_time = start_time
    	self.stop_time = stop_time
        self.duration = start_time - stop_time
    
    def set_timestep(self, interval_seconds):
    	if self.duration == None:
    		return
        self.interval_seconds = interval_seconds
        self.num_timesteps = floor(self.duration / self.interval_seconds)

	def set_spills(self, coords, num_particles_array, release_time_array):
		if self.map == None:
			return
		map(self.map.set_spill, coords, num_particles_array, release_time_array)
	
	def create_environment(self):
		for spill in self.map.spills:
			tmp_list = numpy.ndarray(spill[1], le_rec)
			release_time = spill[2]
			num_particles
			for i in xrange(0, tmp_list.size):
				tmp_list[i]['p']['p_long'] = spill[0][0]
				tmp_list[i]['p']['p_lat'] =  spill[0][1]
				tmp_list[i]['status_code'] = c_gnome.status_not_released
			self.particles += [(tmp_list, release_time)]
	
	def get_num_timesteps(self):
		return self.num_timesteps
	
	def release_particles(self, time_step):
		to_be_kept = range(0, len(self.particles))
		for j in xrange(0, len[self.particles]):
			if self.particles[j][1] <= self.start_time + self.interval_seconds*time_step:
				to_be_kept.remove(j)
				tmp_list = self.particles[j][0]
				for i in xrange(0, tmp_list.size):
					tmp_list[i]['status_code'] = c_gnome.status_in_water
				self.live_particles += [self.particles[j]]
		self.particles = [self.particles[k] for k in to_be_kept]
				
	def refloat_particles(self, time_step):
		return	
		spills = zip(*self.live_particles)[0]
		map(self.map.agitate_particles, [time_step]*len(spills), spills)
		
	def move_particles(self, time_step):
		spills = zip(*self.live_particles)[0]
		for mover in self.movers:
			map(mover.get_move, [time_step]*len(spills), spills)
				
	def step(self, time_step):
		if(self.duration == None):
			return
		if(self.interval_seconds == None):
			return
		if not len(movers) > 0:
			return
		release_particles(time_step)
		refloat_particles(time_step)
		move_particles(time_step)

	def get_particles(self):
		return self.particles