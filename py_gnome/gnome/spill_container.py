#!/usr/bin/env python

"""
spill_container.py

Impliments a container for spills -- keeps all the data from each spill in one set of arrays

The spills themselves provide the arrays themselves (adding more each time LEs are released)

This is the "magic" class -- it handles the smart allocation of arrays, etc.

"""
import datetime

import numpy as np

from gnome import basic_types
from gnome.utilities import rand    # not to confuse with python random module
import gnome.spill
from gnome.utilities.orderedcollection import OrderedCollection


class SpillContainer(object):
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
        
        self.is_uncertain = uncertain   # uncertainty spill - same information as basic_types.spill_type
        self.is_active = True       # sets whether the spill is active or not
        
        self.spills = OrderedCollection(dtype=gnome.spill.Spill)
        self.reset()


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
    @property
    def num_elements(self):
        """
        The number of elements currently in the SpillContainer
        """
        return len(self['positions']) # every spill should have a postitions data array

    def copy(self, uncertain=False):
        import copy
        new_sc = copy.deepcopy(self)
        #new_sc = SpillContainer()
        new_sc.is_uncertain = uncertain

        return new_sc

    # def add_spill(self, spill):
    #     self.spills.add(spill)

    # def remove_spill(self, spill):
    #     """
    #     remove the given spill from the collection

    #     :param spill: the spill object to remove
    #     """
    #     self.spills.remove(spill)

    # def remove_spill_by_id(self, spill_id):
    #     """
    #     remove the spill that has the given id

    #     :param id: the id of the spill you want to remove
    #     """

    #     for spill in self.spills:
    #         if spill.id == spill_id:
    #             self.spills.remove(spill)
    #             break

    
    # def get_spill(self, id):
    #     """
    #     return the spill with a given id
        
    #     :param id: the id of the spill desired

    #     returns None if there is no spill with that id
    #     """
    #     # fixme: used an ordered_dict for efficiency?
    #     for spill in self.spills:
    #         if spill.id == id:
    #             return spill
    #     return None

    def reset(self):
        """
        resets all the spills and stored arrays are cleared, then replaced with
        appropriate empty arrays
        """
        for spill in self.spills:
            spill.reset()
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
                for name, array in new_data.items():
                    if name in self._data_arrays:
                        self._data_arrays[name] = np.r_[ self._data_arrays[name], new_data[name] ]
                    else:
                        self._data_arrays[name] = new_data[name]

    def update_windage(self, time_step):
        ##fixme: this really doesn't seem to belonge here.
        ##       regardless, we need to know the windage_ranges....
        """
        Update windage for each LE for each time step
        May want to cythonize this to speed it up
        """
        self['windages'][:] = rand.random_with_persistance(self.windage_range[0],
                                                          self.windage_range[1],
                                                          self.windage_persist,
                                                          time_step,
                                                          array_len=self.num_LEs)


    def __str__(self):
        msg = "gnome.spill_container.SpillContainer\nspill LE attributes: %s"%self._data_arrays.keys()
        return msg

    __repr__ = __str__ # should write a better one, I suppose

        

class TestSpillContainer(SpillContainer):
    """
    A really simple spill container, pre-initialized with LEs at a point.

    This make sit easy to use for tesing other classes -- movers, maps, etc.
    """
    def __init__(self,
                 num_elements=0,
                 start_pos=(0.0,0.0,0.0),
                 release_time=datetime.datetime(2000,1,1,1),
                 uncertain=False):
        """
        initilize a simple spill container (instantaneous point release)
        """
        SpillContainer.__init__(self, uncertain=uncertain)

        spill = gnome.spill.SurfaceReleaseSpill(num_elements,
                                                start_pos,    
                                                release_time)
        self.spills.add(spill)
        self.release_elements(release_time)




