#!/usr/bin/env python

"""
spill2.py

a new implimenation of the spill class(s)

keeps all the data in separate arrays, so we only store and move around the data that is needed

"""



class FloatingSpill(object):
    """
    An example of a class that contains all we need to know about a spill.

    This is the simplest class -- just a point release of floating
    non-weathering particles

    Many of the "fields" associated with a collection of LEs are optional,
    or used only by some movers, so only the ones required will be requested
    by each mover.
    """
    def __init__(self, num_LEs, start_position, release_time, windage=(0.01, 0.04)):
        """
        param: num_LEs: number of LEs used for this spill
        param: start_position: location the LEs are released (long, lat) (floating point)
        param: release_time: time the LEs are released (datetime object)
        param: windage: the windage range of the LEs (min, max) -- default is (0.01, 0.04) --1% to 4%
        param: uncertain: is this an Uncertainty spill? (True or False, default: False)
        """
        self.num_LEs = numLEs
        self.release_time = release_time
        self.start_position = start_position
        self.windage = windage

        # the core dict of all the arrays:
        self.data_dict = {}
        
        self.__init_LEs()

    def __init_LEs(self):
        """
        called internally to initialize the data arrays needed
        """
        ##fixme: perhaps the data types should be pulled from a central location, 
        ##       to make sure it's compatible with the the C++ types. 
        self.data_dict['positions'] = np.ndarray((num_particles, 2), dtype=np.float64)
        self.data_dict['windage'] =  np.ndarray((num_particles, ), dtype = np.float64)
        self.data_dict['last_water_pt'] = np.ndarray((num_particles, 2), dtype = np.float64)
        self.data_dict['status_code'] = np.ndarray((num_particles, ), dtype = np.int16)

    def get_data_arrays(self, fields):
        """
        get the data arrays that are required
        
        param: fields: a list of the fields that you want
        
        a dict of the data, indexed by the field names, is returned
        """
        data_arrays = {}
        for key in fields:
            data_arrays[key] = self.data_dict[key]
        
        return data_arrays

## fixme -- is there a need for this, or should we use a flag in the regular
##          version instead?
class InstantaneousSpillUncert(FloatingSpill):
    """
    The "uncertainty" version of a Floating Spill
    """  
    def __init__ (self, *args, **kwargs):
        """
        same __init__ as the FloatingSpill
        """
        FloatingSpill.___init__(self, *args, **kwargs)
        
        # what other parameters do we need here?
        
        