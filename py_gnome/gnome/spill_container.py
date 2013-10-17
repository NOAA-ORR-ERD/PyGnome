
"""
spill_container.py

Implements a container for spills -- keeps all the data from each spill in one
set of arrays. The spills themselves provide some of the arrays themselves
(adding more each time LEs are released).
"""
import numpy as np

import gnome.spill
from gnome.utilities.orderedcollection import OrderedCollection
from gnome import basic_types
import gnome.array_types


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
        :param data_arrays=None: A dict of all the data arrays you want to hold
                                 NOTE: no error checking! they should be
                                       correctly aligned, etc.

        The common use-case for this is for loading from cache for
        re-rendering, etc.
        """
        # uncertainty spill - same information as basic_types.spill_type
        self.uncertain = uncertain
        self.on = True       # sets whether the spill is active or not

        if not data_arrays:
            data_arrays = {}
        self._data_arrays = data_arrays
        self.current_time_stamp = None

        # following internal variable is used when comparing two SpillContainer
        # objects. When testing the data arrays are equal, use this tolerance
        # with numpy.allclose() method. Default is to make it 0 so arrays must
        # match exactly. This will not be true when state is stored midway
        # through the run since positions are stored as single dtype as opposed
        # to double
        self._array_allclose_atol = 0

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

        It will not allow user to add a new data_array - only existing
        data_arrays can be modified.
        All data_arrays are defined in prepare_for_model_run
        """

        array = np.asarray(array)

        if data_name in self._data_arrays:
            # if the array is already here, the type should match
            if array.dtype != self._data_arrays[data_name].dtype:
                raise ValueError("new data array must be the same type")
            # and the shape should match
            if array.shape != self._data_arrays[data_name].shape:
                msg = "data array must be the same shape as original array"
                raise ValueError(msg)

        else:
            # make sure length(array) equals length of other data_arrays.
            # check against one key
            if array.shape == ():
                raise TypeError("0-rank arrays are not valid. "\
                               "If new data is a scalar, enter a list [value]")

            if (len(array) !=
                len(self._data_arrays[self._data_arrays.keys()[0]])):
                raise IndexError("length of new data should match length of"\
                                 " existing data_arrays.")

        self._data_arrays[data_name] = array

    def __eq__(self, other):
        """
        Compare equality of two SpillContanerData objects
        """
        if type(self) != type(other):
            return False

        if len(self.__dict__) != len(other.__dict__):
            return False

        # check key/val that are not dicts
        val_is_dict = []
        for key, val in self.__dict__.iteritems():
            """ compare dict not including _data_arrays """
            if isinstance(val, dict):
                val_is_dict.append(key)

            elif val != other.__dict__[key]:
                return False

        # check key, val that are dicts
        for item in val_is_dict:
            if len(self.__dict__[item]) != len(other.__dict__[item]):
                # dicts should contain the same number of keys,values
                return False

            for key, val in self.__dict__[item].iteritems():
                if isinstance(val, np.ndarray):
                    # np.allclose will not work for scalar array so when key is
                    # current_time_stamp, need to do something else
                    if len(val.shape) == 0:
                        if val != other.__dict__[item][key]:
                            return False
                    else:
                        # we know it is an array, not a scalar in an
                        # array - allclose will work
                        if not np.allclose(val, other.__dict__[item][key],
                                           0, self._array_allclose_atol):
                            return False
                else:
                    if val != other.__dict__[item][key]:
                        return False

        return True

    def __ne__(self, other):
        """
        Compare inequality (!=) of two SpillContanerData objects
        """
        if self == other:
            return False
        else:
            return True

    @property
    def num_released(self):
        """
        The number of elements currently in the SpillContainer

        If SpillContainer is initialized, all data_arrays exist even if no
        elements are released so this will always return a valid int >= 0

        This only returns None for SpillContainerData object is initialized
        without any data_arrays.

        todo: Will we ever return None?
        """
        if self._data_arrays.keys():
            return len(self[self._data_arrays.keys()[0]])
        else:
            # should never be the case
            return None

    @property
    def data_arrays_dict(self):
        """
        Returns a dict of the all the data arrays
        """
        # this is a property in case we want change the internal implementation
        return self._data_arrays


class SpillContainer(SpillContainerData):
    """
    Container class for all spills -- it takes care of capturing the released
    LEs from all the spills, putting them all in a single set of arrays.

    Many of the "fields" associated with a collection of elements are optional,
    or used only by some movers, so only the ones required will be requested
    by each mover.

    The data for the elements is stored in the _data_arrays dict. They can be
    accessed by indexing. For example:

    positions = spill_container['positions'] : returns a (num_LEs, 3) array of
    world_point_types
    """

    def __init__(self, uncertain=False):
        super(SpillContainer, self).__init__(uncertain=uncertain)
        self.spills = OrderedCollection(dtype=gnome.spill.Spill)

        # create a new dict from _array_types.SpillContainer
        # This is so original dict is not updated if we update this dict
        # However, note that updating the values in this dict will change
        # original, since the ArrayType objects are mutable
        self._array_types = dict(gnome.array_types.SpillContainer)
        self.rewind()

    def __setitem__(self, data_name, array):
        """
        Invoke baseclass __setitem__ method so the _data_array is set correctly

        In addition, create the appropriate ArrayType if it wasn't created by
        the user.
        """
        super(SpillContainer, self).__setitem__(data_name, array)
        if data_name not in self._array_types:
            shape = self._data_arrays[data_name].shape[1:]
            dtype = self._data_arrays[data_name].dtype.type
            self._array_types[data_name] = gnome.array_types.ArrayType(shape,
                                                                       dtype)

    @property
    def array_types(self):
        # copy internal dict object and return to user. The values in the dict
        # are ArrayType objects - they are mutable and their properties can
        # be changed.
        # Use prepare_for_model_run() to add key, values in the beginning but
        # not in middle of run.
        # Default key, values defined in array_types.SpillContainer cannot
        # be added/deleted.
        return dict(self._array_types)

    def rewind(self):
        """
        In the rewind operation, we:
        - rewind all the spills
        - restore array_types to contain only array_types.SpillContainer
          - movers/weatherers could have been deleted and we don't want to
            carry associated data_arrays
          - prepare_for_model_run() will be called before the next run and
            new arrays can be given

        - purge the data arrays
          - we gather data arrays for each contained spill
          - the stored arrays are cleared, then replaced with appropriate
            empty arrays
        """
        for spill in self.spills:
            spill.rewind()
        # create a full set of zero-sized arrays. If we rewound, something
        # must have changed
        if self._array_types != gnome.array_types.SpillContainer:
            self._array_types = dict(gnome.array_types.SpillContainer)

        self.initialize_data_arrays()

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

    def prepare_for_model_run(self, current_time, array_types={}):
        """
        called when setting up the model prior to 1st time step
        This is considered 0th timestep by model

        Make current_time optional since SpillContainer doesn't require it
        especially for 0th step; however, the model needs to set it because
        it will write_output() after each step. The data_arrays along with
        the current_time_stamp must be set in order to write_output()
        """
        self.current_time_stamp = current_time

        # Question - should we purge any new arrays that were added in previous
        # call to prepare_for_model_run()?
        # I think so, we should start w/ a clean state
        if self._array_types != gnome.array_types.SpillContainer:
            self._array_types = dict(gnome.array_types.SpillContainer)
        self._array_types.update(array_types)

        # define all data arrays before the run begins even if dict is not
        # empty. No need to keep arrays for movers that were deleted.
        # Deletion of mover does not cause a rewind.
        # For the case when model is rewound, then a mover is deleted. The
        # associated data_arrays should be removed.
        self.initialize_data_arrays()

    def prepare_for_model_step(self, current_time):
        """
        Called at the beginning of a time step
        set the current_time_stamp attribute
        """
        self.current_time_stamp = current_time

    def initialize_data_arrays(self):
        """
        initialize_data_arrays() is called without input data during rewind
        and prepare_for_model_run to define all data arrays. At this time the
        arrays are empty.
        """
        for name, elem in self._array_types.iteritems():
            # Initialize data_arrays with 0 elements
            self._data_arrays[name] = elem.initialize_null()

    def _append_data_arrays(self, num_released):
        """
        initialize data arrays once spill has spawned particles
        Data arrays are set to their initial_values

        :param spill_arrays: numpy arrays for 'position' and 'mass'
            _array_types. These are returned by release_elements() of spill
            object
        :param spill: Spill which released the spill_arrays

        """

        for name, array_type in self._array_types.iteritems():
            # initialize all arrays even if 0 length
            self._data_arrays[name] = np.r_[self._data_arrays[name],
                                    array_type.initialize(num_released)]

    def release_elements(self, current_time, time_step):
        """
        Called at the end of a time step

        This calls release_elements on all of the contained spills, and adds
        the elements to the data arrays
        """

        for spill in self.spills:
            if spill.on:
                num_released = spill.num_elements_to_release(current_time,
                                                             time_step)
                if num_released > 0:
                    # update 'spill_num' ArrayType's initial_value so it
                    # corresponds with spill number for this set of released
                    # particles - just another way to set value of spill_num
                    # correctly
                    self._array_types['spill_num'].initial_value = \
                                    self.spills.index(spill.id,
                                                      renumber=False)
                    # unique identifier for each new element released
                    # this adjusts the _array_types initial_value since the
                    # initialize function just calls:
                    #  range(initial_value, num_released + initial_value)
                    self._array_types['id'].initial_value = \
                        len(self['spill_num'])

                    # append to data arrays
                    self._append_data_arrays(num_released)
                    spill.set_newparticle_values(num_released, current_time,
                                                 time_step, self._data_arrays)

    def model_step_is_done(self):
        """
        Called at the end of a time step
        Need to remove particles marked as to_be_removed...
        """
        if len(self._data_arrays) == 0:
            return  # nothing to do - arrays are not yet defined.
        to_be_removed = np.where(self['status_codes'] ==
                                 basic_types.oil_status.to_be_removed)[0]
        if len(to_be_removed) > 0:
            for key in self._array_types.keys():
                self._data_arrays[key] = np.delete(self[key], to_be_removed,
                                                   axis=0)

    def __str__(self):
        msg = ("gnome.spill_container.SpillContainer\nspill LE attributes: %s"
               % self._data_arrays.keys())
        return msg

    __repr__ = __str__


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
        self._spill_container = sc

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

    #LE_data = property(lambda self: self._spill_container._data_arrays.keys())
    @property
    def LE_data(self):
        data = self._spill_container._data_arrays.keys()
        data.append('current_time_stamp')
        return data

    def LE(self, prop_name, uncertain=False):
        if uncertain:
            if prop_name == 'current_time_stamp':
                return self._u_spill_container.current_time_stamp

            return self._u_spill_container[prop_name]
        else:
            if prop_name == 'current_time_stamp':
                return self._spill_container.current_time_stamp

            return self._spill_container[prop_name]

    def __eq__(self, other):
        """
        Compare equality of two SpillContainerPairData objects
        """
        if type(self) != type(other):
            return False

        if self.uncertain != other.uncertain:
            return False

        for sc in zip(self.items(), other.items()):
            if sc[0] != sc[1]:
                return False

        return True

    def __ne__(self, other):
        """
        Compare inequality (!=) of two SpillContainerPairData objects
        """
        if self == other:
            return False
        else:
            return True


class SpillContainerPair(SpillContainerPairData):
    """
    Container holds two SpillContainers, one contains the certain spills while
    the other contains uncertainty spills if model uncertainty is on.
    """
    def __init__(self, uncertain=False):
        """
        initialize object:
        init spill_container, _uncertain and u_spill_container if uncertain

        Note: all operations like add, remove, replace and __iter__ are exposed
        to user for the spill_container.spills OrderedCollection
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
        info = ("{0.__class__},\n  uncertain={0.uncertain}\n  Spills: {1}"
                .format(self, self._spill_container.spills))
        return info

    @property
    def uncertain(self):
        return self._uncertain

    @uncertain.setter
    def uncertain(self, value):
        if type(value) is not bool:
            raise TypeError("uncertain property must be a bool (True/False)")

        if self._uncertain == True and value == False:
            self._uncertain = value
            del self._u_spill_container  # delete if it exists
            self.rewind()  # Not sure if we want to do this?
        elif self._uncertain == False and value == True:
            self._uncertain = value
            self._u_spill_container = self._spill_container.uncertain_copy()
            self.rewind()

    def add(self, spill):
        """
        add spill to spill_container and make copy in u_spill_container if
        uncertainty is on

        Overload add method so it can take a tuple (spill, uncertain_spill)
        """
        if isinstance(spill, tuple):
            if self.uncertain:
                if len(spill) != 2:
                    raise ValueError("You can only add a tuple containing a"\
                                     " certain/uncertain spill pair"\
                                     " (spill, uncertain_spill)")
                self._u_spill_container.spills += spill[1]
            else:
                if len(spill) != 1:
                    raise ValueError("Uncertainty is off. Tuple must only"\
                                     " contain (certain_spill,)")

            self._spill_container.spills += spill[0]

        else:
            self._spill_container.spills += spill
            if self.uncertain:
                self._u_spill_container.spills += spill.uncertain_copy()

    def remove(self, ident):
        """
        remove object from spill_container.spills and the corresponding
        uncertainty spill as well
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
        The uncertain and certain spill containers will contain the same number
        of spills return the length of spill_container.spills
        """
        return len(self._spill_container.spills)

    def __contains__(self, ident):
        """
        looks to see if ident which is the id of a spill belongs in the
        _spill_container.spills OrderedCollection
        """
        return ident in self._spill_container.spills

    def to_dict(self):
        """
        takes the instance of SpillContainerPair class and outputs a dict with:
            'certain_spills': call to_dict() on spills ordered collection
            stored in certain spill container

        if uncertain, then also return:
            'uncertain_spills': call to_dict() on spills ordered collection
            stored in uncertain spill container
        """
        dict_ = {'certain_spills': self._spill_container.spills.to_dict()}
        if self.uncertain:
            dict_.update({'uncertain_spills':
                          self._u_spill_container.spills.to_dict()})

        return dict_
