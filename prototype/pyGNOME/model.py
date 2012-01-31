
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
        self.duration = None
        self.timestep = None
        self.num_timesteps = None
        
    def add_mover(self, type):
    	pass
    	
    def add_map(self, image_size, bna_filename):
        self.map = map(image_size, bna_filename)
    
    def add_wind_mover(self, constant_wind_value):
        movers += [c_gnome.wind_mover(constant_wind_value)]
        
    def add_random_mover(self, diffusion_coefficient):
        movers += [c_gnome.random_mover(diffusion_coefficient)]
        
    def set_run_duration(self, run_duration_seconds):
        self.duration = run_duration_seconds
    
    def set_timestep(self, interval_seconds):
        self.timestep = interval_seconds
        self.num_timesteps = floor(self.duration / self.timestep)

	def set_spills(self, coords):
		if self.map == None:
			return
		map(self.map.set_spill, coords)
	
	def create_environment(self):
		for spill in self.map.spills:
			tmp_list = numpy.ndarray(spill[1], le_rec)
			for i in xrange(0, tmp_list.size):
				tmp_list[i]['p']['p_long'] = spill[0][0]
				tmp_list[i]['p']['p_lat'] =  spill[0][1]
			self.particles += [tmp_list]
	
	def get_num_timesteps(self):
		return self.num_timesteps
		
	def move_particles(self, time_step):	
		for mover in self.movers:
			map(mover.get_move, [time_step]*len(self.particles), self.particles)
				
	def step(self, time_step):
		if(self.duration == None):
			return
		if(self.timestep == None):
			return
		if not len(movers) > 0:
			return
		
		move_particles(time_step)

	def get_particles(self):
		return self.particles