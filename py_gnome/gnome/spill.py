from basic_types import le_rec, status_not_released, status_in_water, status_on_land, disp_status_have_dispersed, world_point_type
from math import ceil, pow
from random import random
import numpy

class spill:

    """
    Spill container. Currently handles continuous release, as well as refloating.
    (I suspect that refloating needs some more attention.)
    """

    def __init__(self, gnome_map, num_particles, disp_status, windage, \
                    (start_time, stop_time), (start_position, stop_position),uncertain=False):
        """
            Initializes spill attributes.
        ++args:
            gnome_map must be a pyGNOME land-water map, or one of a similar implementation.
            disp_status is the spill's initial status (see gnome.basic_types for more information)
            windage has units? I suspect not
            (start_time, stop_time) must be in seconds
            (start_position, stop_position) is a tuple of tuples, in lat-lon coordinates
            uncertain's value determines whether this spill will be treated for Minimum Regret.
        """
        self.npra = numpy.ndarray(num_particles, dtype=le_rec)
        self.num_particles = num_particles
        self.start_time = start_time
        self.stop_minus_start_time = stop_time - start_time
        self.start_position = start_position
        self.stop_minus_start_pos = (stop_position[0] - start_position[0], stop_position[1] - start_position[1])
        self.gnome_map = gnome_map
        self.released_index = 0
        self.windage = windage
        self.uncertain = uncertain
        self.initialize_spill(disp_status)
        self.chromgph = None
        
    def initialize_spill(self, disp_status):

        """ Initializes the spill's numpy array. See gnome.basic_types for more information. """
        self.npra['status_code'] = status_not_released
        self.npra['dispersion_status'] = disp_status
        self.npra['p']['p_long'] = self.start_position[0]
        self.npra['p']['p_lat'] = self.start_position[1]
        self.npra['windage'] = self.windage
        self.npra['z'] =  0.
    
    def release_particles(self, model_time):
        """ 
            Depending on the current model time, releases particles that are due to be released and that have not yet been. 
            Simple algorithm that determines the proportion of the spill to be released per unit time, and determines
            the proportion of displacement similarly.
        """
        if(self.released_index >= self.num_particles):
            self.release_particles = lambda null: None
            return
        if(self.stop_minus_start_time == 0):
            fraction_duration = 1
        else:
            fraction_duration = ((float(model_time) - self.start_time) / self.stop_minus_start_time)
            if fraction_duration < 0:
                return
        fraction_duration = min(1, fraction_duration)
        displacement = (fraction_duration * self.stop_minus_start_pos[0], fraction_duration *self.stop_minus_start_pos[1])
        point_of_release = self.start_position + displacement
        ra = self.npra['status_code']
        for self.released_index in xrange(self.released_index, int(ceil(fraction_duration*self.num_particles))):
            ra[self.released_index]=status_in_water
        self.released_index += 1
        
    def refloat_particles(self, length_time_step, lwpra):
        ## this should be in the model or map class
        """ 
            Refloats particles that have been landed.
            Takes into account the proportion of the model's
            configured time step that is taken by the land-water
            map's configured half-life, in order to determine
            roughly how many half-lives have expired in a single
            step of the model. Does not currently support hindcaasting.
        """
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
        ## fixme: what is this?
        pass
    
    def noneTypePrevention(self, chromgph):
        ## fixme: what is this for?
        self.chromgph = chromgph
        self.noneTypePrevention = lambda null: None

## moved to map.py        
#    def movement_check(self):
#        """ 
#        After moving a spill (after superposing each of the movers' contributions),
#        we determine which of the particles have been landed, ie., that are in the
#        current time step on land but were not in the previous one. Chromgph is a list
#        of boolean values, determining which of the particles need treatment. Particles
#        that have been landed are beached.
#        """
#        ##fixme: this code should be in the map classes
#        
#        # make a regular Nx2 numpy array 
#        coords = numpy.copy(self.npra['p']).view(world_point_type).reshape((-1, 2),)
#        coords = self.gnome_map.to_pixel_array(coords)
#        chromgph = self.gnome_map._on_land_pixel_array(coords)
#        #chromgph = map(self.gnome_map.on_land_pixel, coords)
#        sra = self.npra['status_code']
#        for i in xrange(0, self.num_particles):
#            if chromgph[i]:
#                sra[i] = status_on_land
#        if self.chromgph == None:
#            self.chromgph = chromgph
#        merg = [int(chromgph[x] and not self.chromgph[x]) for x in xrange(0, len(chromgph))]
#        self.chromgph = chromgph
#        return merg
