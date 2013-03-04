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
    def __init__(self, data_arrays=None, uncertain=False):
        """
        Initialize a SimpleSpillContainer.

        :param uncertain=False: flag indicating whether this holds uncertainty
                                elements or not 
        :param data_arrays=None: A dict of all the data arrays you want to hold.
                                 NOTE: no error checking! they should be correctly
                                       aligned, etc.

        The common use-case for this is for loading from cache for re-rendering, etc.
        """
        self.uncertain = uncertain   # uncertainty spill - same information as basic_types.spill_type
        self.on = True       # sets whether the spill is active or not
        
        if not data_arrays:
            data_arrays = {}
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
        super(SpillContainer, self).__init__(uncertain=uncertain)
        
        self.all_array_types = {}
        self.spills = OrderedCollection(dtype=gnome.spill.Spill)
        self.rewind()

    def rewind(self):
        """
        In the rewind operation, we:
        - rewind all the spills
        - purge the data arrays
          - we gather data arrays for each contained spill
          - the stored arrays are cleared, then replaced with appropriate empty arrays
        """
        for spill in self.spills:
            spill.rewind()
        # this should create a full set of zero-sized arrays
        # it creates a temporary Spill object, that should reflect
        # the arrays types of all existing Spills
        self._data_arrays = gnome.spill.Spill().create_new_elements(0)

    def reconcile_data_arrays(self):
        self.update_all_array_types()

        # if a spill was added with new properties, we need to
        # create the new property and back-fill the array
        for name, dtype in self.all_array_types.iteritems():
            if name not in self._data_arrays:
                array_type = dict( ((name, dtype),) )
                data_arrays = gnome.spill.Spill().create_new_elements(self.num_elements, array_type)
                self._data_arrays[name] = data_arrays[name]

        # if a spill was deleted, it may have had properties
        # that are not needed anymore
        for k in self._data_arrays.keys()[:]:
            if k not in self.all_array_types:
                del self._data_arrays[k]

    def update_all_array_types(self):
        self.all_array_types = {}
        for spill in self.spills:
            self.all_array_types.update(spill.array_types)

    def get_spill_mask(self, spill):
        return self['spill_num'] == self.spills.index(spill.id)

    def uncertain_copy(self):
        """
        Returns a copy of the spill_container suitable for uncertainty

        It has all the same spills, with the same ids, and the uncertain
        flag set to True
        """
        u_sc = SpillContainer(uncertain=True)
        for sp in self.spills:
            u_sc.spills += sp.uncertain_copy()
        return u_sc

    def prepare_for_model_step(self, current_time, time_step=None):
        """
        Called at the beginning of a time step
        Not sure what might need to get done here...        
        """
        pass

    def release_elements(self, current_time, time_step):
        """
        Called at the end of a time step

        This calls release_elements on all of the contained spills, and adds
        the elements to the data arrays
        """
        self.reconcile_data_arrays()

        for spill in self.spills:
            if spill.on:
                new_data = spill.release_elements(current_time,
                                                  time_step=time_step,
                                                  array_types=self.all_array_types)
                if new_data is not None:
                    if 'spill_num' in new_data:
                        new_data['spill_num'][:] = self.spills.index(spill.id, renumber=False)
                    for name in new_data:
                        if name in self._data_arrays:
                            self._data_arrays[name] = np.r_[ self._data_arrays[name], new_data[name] ]
                        else:
                            self._data_arrays[name] = new_data[name]

    def __str__(self):
        msg = "gnome.spill_container.SpillContainer\nspill LE attributes: %s" % self._data_arrays.keys()
        return msg

    __repr__ = __str__  # should write a better one, I suppose


class SpillContainerPairData(object):
    """
    A really simple SpillContainerPair
      - holds SpillContainerPairData objects,
        but doen't manage spills, etc.
    
    Think of it as a read-only SpillContainerPair.

    Designed primarily to hold data retrieved from cache

    """
    def __init__(self, sc, u_sc=None):
        """
        initialize object with the spill_containers passed in.
        """
        
        if sc.uncertain:
            raise ValueError("sc is an uncertain SpillContainer")
        self._spill_container = sc  # name mangling just to make it more difficult for user to find
        
        if u_sc is None:
            self._uncertain = False
        else:
            self._uncertain = True
            if not u_sc.uncertain:
                raise ValueError("u_sc is not an uncertain SpillContainer")
            self._u_spill_container = u_sc

    def __repr__(self):
        """
        unambiguous repr
        """
        info = "{0.__class__},\n  uncertain={0.uncertain}\n ".format(self)
        return info

    @property
    def uncertain(self):
        return self._uncertain

    def items(self):
        """
        returns a tuple of the enclosed spill containers

        if uncertainty is off, just one is in the tuple
        if uncertainly is on -- then it is a two-tuple:
            (certain_container, uncertain_container)

        To act on both:
            for sc in spill_container_pair.items():
                do_something_with(sc)
        """
        ## NOTE: cache code counts on the uncertain SpillContainer being last
        if self.uncertain:
            return (self._spill_container, self._u_spill_container)
        else:
            return (self._spill_container,)

    LE_data = property(lambda self: self._spill_container._data_arrays.keys())    
    
    def LE(self, prop_name, uncertain=False):
        if uncertain:
            return self._u_spill_container[prop_name]
        else:
            return self._spill_container[prop_name]



class SpillContainerPair(SpillContainerPairData):
    """
    Container holds two SpillContainers, one contains the certain spills while the other contains
    uncertainty spills if model uncertainty is on.
    """
    def __init__(self, uncertain=False):
        """
        initialize object: 
        init spill_container, _uncertain and u_spill_container if uncertain

        Note: all operations like add, remove, replace and __iter__ are exposed to user
        for the spill_container.spills OrderedCollection

        Since spill_container.spills are 
        """
        sc = SpillContainer()
        if uncertain:
            u_sc = SpillContainer(uncertain=True)
        else:
            u_sc = None

        super(SpillContainerPair, self).__init__(sc, u_sc)

    def rewind(self):
        """
        rewind spills in spill_container
        """
        self._spill_container.rewind()
        if self.uncertain:
            self._u_spill_container.rewind()

    def __repr__(self):
        """
        unambiguous repr
        """
        info = "{0.__class__},\n  uncertain={0.uncertain}\n  Spills: {1}".format(self, self._spill_container.spills)
        return info

    @property
    def uncertain(self):
        return self._uncertain

    @uncertain.setter
    def uncertain(self, value):
        if type(value) is not bool:
            raise TypeError("uncertain property must be a bool (True/False)")
        if self._uncertain == True and value == False:
            del self._u_spill_container  # delete if it exists
            self.rewind()  # Not sure if we want to do this?
        elif self._uncertain == False and value == True:
            self._u_spill_container = self._spill_container.uncertain_copy()
            self.rewind()

        self._uncertain = value

    def add(self, spill):
        """
        add spill to spill_container and make copy in u_spill_container if uncertainty is on
        """
        self._spill_container.spills += spill
        spill.spill_num.initial_value = self._spill_container.spills.index(spill.id, renumber=False)
        if self.uncertain:
            # todo: make sure spill_num for copied spill are the same as original
            self._u_spill_container.spills += spill.uncertain_copy()

    def remove(self, ident):
        """
        remove object from spill_container.spills and the corresponding uncertainty spill
        as well
        """
        if self.uncertain:
            idx = self._spill_container.spills.index(ident)
            u_ident = [s for s in self._u_spill_container.spills][idx]
            del self._u_spill_container.spills[u_ident.id]
        del self._spill_container.spills[ident]

    def __getitem__(self, ident):
        """
        only return the certain spill
        """
        spill = self._spill_container.spills[ident]
        return spill

    def __delitem__(self, ident):
        self.remove(ident)

    def __iadd__(self, rop):
        self.add(rop)
        return self

    def __iter__(self):
        """
        iterates over the spills defined in spill_container
        """
        for sp in self._spill_container.spills:
            yield self.__getitem__(sp.id)

    def __len__(self):
        """        
        It refers to the total number of spills that have been added
        The uncertain and certain spill containers will contain the same number of spills
        return the length of spill_container.spills
        """
        return len(self._spill_container.spills)    

    def __contains__(self, ident):
        """
        looks to see if ident which is the id of a spill belongs in the _spill_container.spills 
        OrderedCollection
        """
        return ident in self._spill_container.spills


class TestSpillContainer(SpillContainer):
    """
    A really simple spill container, pre-initialized with LEs at a point.

    This makes it easy to use for tesing other classes -- movers, maps, etc.
    """
    def __init__(self,
                 num_elements=0,
                 start_pos=(0.0, 0.0, 0.0),
                 release_time=datetime.datetime(2000, 1, 1, 1),
                 uncertain=False):
        """
        initilize a simple spill container (instantaneous point release)
        """
        super(TestSpillContainer, self).__init__(uncertain=uncertain)

        spill = gnome.spill.SurfaceReleaseSpill(num_elements,
                                                start_pos,
                                                release_time)
        spill.spill_num.initial_value = 0
        self.spills.add(spill)
        self.release_elements(release_time, 360)


