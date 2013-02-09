#!/usr/bin/env python

"""
spill_container.py

Impliments a container for spills -- keeps all the data from each spill in one set of arrays

The spills themselves provide the arrays themselves (adding more each time LEs are released)

This is the "magic" class -- it handles the smart allocation of arrays, etc.

"""
import datetime

import numpy as np

import gnome.spill
from gnome.utilities.orderedcollection import OrderedCollection

class SpillContainerData(object):
    """
    A really simple SpillContainer -- holds the data arrays,
    but doesn't manage spills, etc.
    
    Think of it as a read-only SpillContainer.

    Designed primarily to hold data retrieved from cache

    """
    def __init__(self, uncertain=False, data_arrays=None):
        """
        Initilize a SimpleSpillContainer.

        :param uncertain=False: flag indicating whether this holds uncertainty
                                elements or not 
        :param data_arrays=None: A dict of all the data arrays you want to hold.
                                 NOTE: no error checking! theyshould be correctly
                                       aligned, etc.
        """
        print "in SpillContainerData.__init__"
        print uncertain
        
        self.is_uncertain = uncertain   # uncertainty spill - same information as basic_types.spill_type
        self.on = True       # sets whether the spill is active or not
        
        if data_arrays is not None:
            self._data_arrays = data_arrays

    def __getitem__(self, data_name):
        """
        The basic way to access data for the LEs
        
        :param data_name: the name of the array to be returned

        example:  a_spill_container['positions'] give you the
                  (x,y,z positions array of the elements)

        :raises KeyError: raised if the data is not there
        """
        return self._data_arrays[data_name]
    
    def __setitem__(self, data_name, array):
        
        """
        sets the data item
        
        careful! -- this should probably only be used for testing!
        as all arrays need to be compatible
        
        It will be checked to at least be size-consistent with the rest of the
        data, and type-consistent if the data array is being replaced
        """

        array = np.asarray(array)
        
        if data_name in self._data_arrays:
            # if the array is already here, the type should match        
            if array.dtype !=  self._data_arrays[data_name].dtype:
                raise ValueError("new data array must be the same type")
            # and the shape should match        
            if array.shape !=  self._data_arrays[data_name].shape:
                raise ValueError("new data array must be the same shape")
                    
        self._data_arrays[data_name] = array

    @property
    def num_elements(self):
        """
        The number of elements currently in the SpillContainer
        """
        return len(self['positions']) # every spill should have a postitions data array

    @property
    def data_arrays_dict(self):
        """
        Returns a dict of the all the data arrays 
        """
        ## this is a propery in case we want change the internal implimentation
        return self._data_arrays


class SpillContainer(SpillContainerData):
    """
    Container class for all spills -- it takes care of capturing the released LEs from
    all the spills, putting them all in a single set of arrays.
    
    Many of the "fields" associated with a collection of elements are optional,
    or used only by some movers, so only the ones required will be requested
    by each mover.
    
    The data for the elements is stored in the _data_arrays dict. They can be
    accessed by indexing. For example:
     
    positions = spill_container['positions'] : returns a (num_LEs, 3) array of world_point_types
    
    """
    def __init__(self, uncertain=False):
        super(SpillContainer, self).__init__(uncertain)
        
        self.spills = OrderedCollection(dtype=gnome.spill.Spill)
        self.rewind()


    def __getitem__(self, data_name):
        """
        The basic way to access data for the LEs
        
        a KeyError will be raised if the data is not there
        """
        return self._data_arrays[data_name]
    
    def __setitem__(self, data_name, array):
        
        """
        sets the data item
        
        careful! -- this should probably only be used for testing!
        as all arrays need to be compatible
        
        It will be checked to at least be size-consistent with the rest of the
        data, and type-consistent if the data array is being replaced
        """

        array = np.asarray(array)
        
        if data_name in self._data_arrays:
            # if the array is already here, the type should match        
            if array.dtype !=  self._data_arrays[data_name].dtype:
                raise ValueError("new data array must be the same type")
            # and the shape should match        
            if array.shape !=  self._data_arrays[data_name].shape:
                raise ValueError("new data array must be the same shape")
                    
        self._data_arrays[data_name] = array

    def uncertain_copy(self):
        """
        Returns a copy of the spill_container suitable for uncertainty

        It has all the same spills, with the same ids, and the is_uncertain
        flag set to True
        """
        u_sc = SpillContainer(uncertain=True)
        for sp in self.spills:
            u_sc.spills += sp.uncertain_copy()
        return u_sc


    def rewind(self):
        """
        rewinds all the spills and the stored arrays are cleared, then replaced with
        appropriate empty arrays
        """
        for spill in self.spills:
            spill.rewind()
        # this should create a full set of zero-sized arrays
        # it creates a temporary Spill object, that should reflect
        # the arrays types of all existing Spills
        self._data_arrays = gnome.spill.Spill().create_new_elements(0)


    def prepare_for_model_step(self, current_time, time_step=None):
        """
        Called at the beginning of a time step

        Note sure what might need to get done here...        
        """
        pass

    def release_elements(self, current_time, time_step=None):
        """
        Called at the end of a time step

        This calls release_elements on all of the contained spills, and adds
        the elements to the data arrays
        
        """
        for spill in self.spills:
            new_data = spill.release_elements(current_time, time_step)
            if new_data is not None:
                for name in new_data:
                    if name in self._data_arrays:
                        self._data_arrays[name] = np.r_[ self._data_arrays[name], new_data[name] ]
                    else:
                        self._data_arrays[name] = new_data[name]



    def __str__(self):
        msg = "gnome.spill_container.SpillContainer\nspill LE attributes: %s"%self._data_arrays.keys()
        return msg

    __repr__ = __str__ # should write a better one, I suppose

        

class TestSpillContainer(SpillContainer):
    """
    A really simple spill container, pre-initialized with LEs at a point.

    This makes it easy to use for tesing other classes -- movers, maps, etc.
    """
    def __init__(self,
                 num_elements=0,
                 start_pos=(0.0,0.0,0.0),
                 release_time=datetime.datetime(2000,1,1,1),
                 uncertain=False):
        """
        initilize a simple spill container (instantaneous point release)
        """
        super(TestSpillContainer, self).__init__(uncertain=uncertain)

        spill = gnome.spill.SurfaceReleaseSpill(num_elements,
                                                start_pos,    
                                                release_time)
        self.spills.add(spill)
        self.release_elements(release_time)




