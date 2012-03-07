""" Documentation string goes here. """

from basic_types import le_rec, status_not_released, status_in_water, status_on_land, disp_status_have_dispersed
from math import ceil, pow
from random import random
import numpy

class spill:

    def __init__(self, gnome_map, num_particles, disp_status, windage, \
                    (start_time, stop_time), (start_position, stop_position),):
        self.npra = numpy.ndarray(num_particles, dtype=le_rec)
        self.num_particles = num_particles
        self.start_time = start_time
        self.stop_minus_start_time = stop_time - start_time
        self.start_position = start_position
        self.stop_minus_start_pos = (stop_position[0] - start_position[0], stop_position[1] - start_position[1])
        self.gnome_map = gnome_map
        self.released_index = 0
        self.windage = windage
        self.initialize_spill(disp_status)
        self.chromgph = None
        
    def initialize_spill(self, disp_status):
        sra = self.npra['status_code']
        dra = self.npra['dispersion_status']
        pra = self.npra['p']
        wra = self.npra['windage']
        zra = self.npra['z']
        for j in xrange(0, self.num_particles):
            sra[j] = status_not_released
            dra[j] = disp_status
            wra[j] = self.windage
            pra[j]['p_long'] = self.start_position[0]
            pra[j]['p_lat'] = self.start_position[1]
            zra[j] = 0
    
    def release_particles(self, model_time):
        if(self.released_index >= self.num_particles):
            self.release_particles = lambda null: None
            return
        if(self.stop_minus_start_time == 0):
            fraction_duration = 1
        else:
            fraction_duration = ((float(model_time) - self.start_time) / self.stop_minus_start_time)
            if fraction_duration < 0:
                return
        displacement = (fraction_duration * self.stop_minus_start_pos[0], fraction_duration *self.stop_minus_start_pos[1])
        point_of_release = self.start_position + displacement
        ra = self.npra['status_code']
        for self.released_index in xrange(self.released_index, min(1, int(ceil(fraction_duration*self.num_particles)))):
            ra[self.released_index]=status_in_water
        self.released_index += 1
        
    def refloat_particles(self, length_time_step, lwpra):
        dra = self.npra['dispersion_status']
        chromgph = self.chromgph
        if chromgph == None:
            return
        considered_indices = []
        for j in xrange(0, self.num_particles):
            if chromgph[j] and not dra[j] == disp_status_have_dispersed: 
                considered_indices += [j]
        refloat_likelihood = 1 - pow(.5, length_time_step/(self.gnome_map.refloat_halflife))
        pra = self.npra['p']
        sra = self.npra['status_code']
        for idx in considered_indices:
            if (random() < refloat_likelihood):
                chromgph[idx] = 0
                sra[idx] = status_in_water
                pra[idx] = lwpra[idx]
                
    def disperse_particles(self):
        pass
    
    def noneTypePrevention(self, chromgph):
        self.chromgph = chromgph
        self.noneTypePrevention = lambda null: None
        
    def movement_check(self):
        coords = numpy.copy(self.npra['p'])
        self.gnome_map.to_pixel_array(coords)
        chromgph = map(self.gnome_map.on_land_pixel, coords)
        sra = self.npra['status_code']
        for i in xrange(0, self.num_particles):
            if chromgph[i]:
                sra[i] = status_on_land
        self.noneTypePrevention(chromgph)
        merg = [int(chromgph[x] and not self.chromgph[x]) for x in xrange(0, len(chromgph))]
        self.chromgph = chromgph
        return merg
