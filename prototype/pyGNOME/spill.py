""" Documentation string goes here. """

from basic_types import le_rec, status_not_released, status_in_water
from math import ceil
import numpy

class spill:

	def __init__(self, map, num_particles, disp_status, \
					(start_time, stop_time), (start_position, stop_position),):
		self.npra = numpy.ndarray(num_particles, le_rec)
		self.num_particles = num_particles
		self.start_time = start_time
		self.stop_minus_start_time = stop_time - start_time
		self.start_position = start_position
		self.stop_minus_start_pos = stop_position - start_position
		self.map = map
		self.released_index = 0
		self.initialize_spill(disp_status)

	def initialize_spill(self, disp_status):
		sra = self.npra['status_code']
		dra = self.npra['dispersion_status']
		for j in xrange(0, self.num_particles):
			sra[j] = status_not_released
			dra[j] = disp_status
			
	
	def do_nothing(self):
		pass
		
	def release_particles(self, model_time):
		if(self.released_index >= self.num_particles):
			self.released_particles = self.do_nothing
		fraction_duration = ((model_time - self.start_time) / self.stop_minus_start_time)
		displacement = fraction_duration * self.stop_minus_start_pos
		point_of_release = self.start_position + displacement
		ra = self.npra['status_code']
		for self.released_index in xrange(self.released_index, ceil(fraction_duration*self.num_particles)):
			ra[self.released_index]=status_in_water
		self.released_index += 1
		
	def refloat_particles(self):
		pass
	
	def movement_check(self):
		pass


