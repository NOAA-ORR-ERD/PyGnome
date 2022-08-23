#!/usr/bin/env python
"""
spill_container.py

Implements a container for spills -- keeps all the data from each spill in one
set of arrays. The spills themselves provide some of the arrays themselves
(adding more each time LEs are released).
"""

import os

import numpy as np

from gnome.basic_types import fate as bt_fate
from gnome.basic_types import oil_status
from gnome.array_types import (default_array_types)

from gnome.utilities.orderedcollection import OrderedCollection
from gnome import AddLogger
import gnome.spills.spill
from gnome.spills.substance import NonWeatheringSubstance


class FateDataView(AddLogger):
    """
    need a docstring -- what is this for?
    """

    _dicts_ = ('surface_weather', 'subsurf_weather', 'skim', 'burn',
               'disperse', 'non_weather', 'all')

    def __init__(self):
        self.reset()

    def reset(self):
        for fate_type in self._dicts_:
            setattr(self, fate_type, {})
        # all data - this is required by WeatheringData to update
        # properties of old LEs and properties of newly released LEs
        self.all = {}

    def _get_fate_mask(self, sc, fate):
        '''
        get fate_status mask over SC - only include LEs with 'mass' > 0.0
        '''
        if fate == 'all':
            # look at all fate data
            w_mask = np.asarray([True] * len(sc))
        else:
            w_mask = (sc['fate_status'] & getattr(bt_fate, fate) == getattr(bt_fate, fate))

        w_mask = np.logical_and(w_mask, sc['mass'] > 0.0)
        return w_mask

    def _set_data(self, sc, array_types, fate_mask, fate_status):
        '''
        Set the data arrays in the FateDataView

        fate_mask is the data already masked for the desired 'fate' option

        fate_status is the status the mask is for ('surface_weather', etc.)
        '''
        # # return all data associated with substance
        # if 'substance' in sc:
        #     fate_mask = np.logical_and(sc['substance'] == self.substance_id,
        #                                fate_mask)

        if np.all(fate_mask):
            # no need to make a copy of array
            setattr(self, fate_status, sc._data_arrays)
        else:
            dict_to_update = getattr(self, fate_status)
            for at in array_types:
                array = sc._array_name(at)

                #if array not in dict_to_update:
                dict_to_update[array] = sc[array][fate_mask]

            setattr(self, fate_status, dict_to_update)

    def get_data(self, sc, array_types, fate_status='surface_weather'):
        '''
        Get data that matches the given fate_status.
        Since this is weathering data, only include elements with 'mass' > 0

        Options are: 'all', 'surface_weather', 'subsurf_weather', 'skim', 'non_weather',
        'burn'
        '''
        self._set_data(sc,
                       array_types,
                       self._get_fate_mask(sc, fate_status),
                       fate_status)
        return getattr(self, fate_status)

    def update_sc(self, sc, fate_status='surface_weather'):
        '''
        update SC arrays with FateDataView arrays for specified fate
        - update all arrays just to make sure everything is in sync

        After update, remove LEs with mass = 0. Since weatherers call this at
        the end of a weathering step, this ensures zero mass LEs are removed
        from the arrays.

        .. note:: the 'id' of each LE corresponds with the index into SC array
                  when it was added. if LEs are removed, then this will not be
                  the case. Do not rely on this indexing. Instead, get the mask
                  again - the assumption is that the fate_mask should be the
                  same between getting the data and resync'ing the original arrays
                  in the SC
        '''
        d_to_sync = getattr(self, fate_status)

        if d_to_sync is sc._data_arrays:
            self.reset()
            #for fs in self._dicts_:
            #    self._set_data( sc, getattr(self, fs).keys(), self._get_fate_mask(sc, fs), fs)
            return

        w_mask = self._get_fate_mask(sc, fate_status)

        # if 'substance' in sc:
        #     w_mask = np.logical_and(sc['substance'] == self.substance_id,
        #                             w_mask)

        # if fate_status of LEs was updated, then reset data attribute. This is
        # because the data in the attribute is no longer valid. For instance,
        # if the 'burn' started with 'surface_weather' data_arrays, then
        # marked some of these LEs to be burned, they should no longer be
        # contained in the 'surface_weather' dict - easiest to reset the dict
        # and let it be recreated when the next weatherer asks for data.
        reset_view = False
        if ('fate_status' in d_to_sync and
                np.any(sc['fate_status'][w_mask] != d_to_sync['fate_status'])):
            reset_view = True
        elif ('mass' in d_to_sync and
              np.any(np.isclose(d_to_sync['mass'], 0))):
            # probably need a threshold close to 0.0 as opposed to equality
            reset_view = True
            self.logger.debug(self._pid + "found LEs with 'mass' equal to 0. "
                              "reset_view")

        for key, val in d_to_sync.items():
            sc[key][w_mask] = val

        if reset_view:
            self.reset()

    def _reset_fatedata(self, sc, ix):
        '''
        reset all arrays that contain LE with 'id' = ix
        '''
        for fate in self._dicts_:
            data = getattr(self, fate)
            if len(data) > 0:
                idx = np.where(data['id'] == ix)[0]
                if len(idx) > 0:
                    self._set_data(sc, data.keys(),
                                   self._get_fate_mask(sc, fate),
                                   fate)


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

        Note: initialize current_time_stamp attribute to None. It is
        responsibility of caller to set current_time_stamp (for eg: Model)
        """
        self.uncertain = uncertain

        # sets whether the spill is active or not
        self.on = True

        if not data_arrays:
            data_arrays = {}

        self._data_arrays = data_arrays
        self.current_time_stamp = None
        self.mass_balance = {}
        self.substance = None

        # following internal variable is used when comparing two SpillContainer
        # objects. When testing the data arrays are equal, use this tolerance
        # with numpy.allclose() method. Default is to make it 0 so arrays must
        # match exactly. This will not be true when _state is stored midway
        # through the run since positions are stored as single dtype as opposed
        # to double
        self._array_allclose_atol = 0

    def __contains__(self, item):
        return item in self._data_arrays

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
                raise ValueError('new data array must be the same type')

            # and the shape should match
            if array.shape != self._data_arrays[data_name].shape:
                msg = 'data array must be the same shape as original array'
                raise ValueError(msg)
        else:
            # make sure length(array) equals length of other data_arrays.
            # check against one key
            if array.shape == ():
                raise TypeError('0-rank arrays are not valid. '
                                'If new data is a scalar, '
                                'enter a list [value]')

            if (len(array) != len(self)):
                raise IndexError('length of new data should match length of '
                                 'existing data_arrays.')

        self._data_arrays[data_name] = array

    def __eq__(self, other):
        'Compare equality of two SpillContanerData objects'
        if type(self) != type(other):
            return False

        if len(self.__dict__) != len(other.__dict__):
            return False

        if self.substance != other.substance:
            return False

        # check key/val that are not dicts
        val_is_dict = []
        for key, val in self.__dict__.items():
            'compare dict not including _data_arrays'
            if isinstance(val, dict):
                val_is_dict.append(key)
            elif key == '_substances_spills' or key == '_fate_data_view':
                '''
                this is just another view of the data - no need to write extra
                code to check equality for this
                '''
                pass
            elif val != other.__dict__[key]:
                return False

        # check key, val that are dicts
        for item in val_is_dict:
            if set(self.__dict__[item]) != set(other.__dict__[item]):
                # dicts should contain the same keys
                return False

            for key, val in self.__dict__[item].items():
                other_val = other.__dict__[item][key]
                if isinstance(val, np.ndarray):
                    try:
                        if not np.allclose(val, other_val, 0,
                                           self._array_allclose_atol):
                            return False
                    except TypeError:
                        # not implemented for this dtype,
                        # so just check equality
                        if not np.all(val == other_val):
                            return False
                else:
                    if val != other_val:
                        return False

        return True

    def __ne__(self, other):
        return not (self == other)

    def __len__(self):
        """
        The "length" of a spill container is the number of elements in it.
        The first dimension of any ndarray in our data_arrays
        will always be the number of elements that are contained in a
        SpillContainer.
        """
        try:
            # find the length of an arbitrary first array
            return len(next(iter(self._data_arrays.values())))
        except StopIteration:
            return 0

    @property
    def num_released(self):
        """
        The number of elements currently in the SpillContainer

        If SpillContainer is initialized, all data_arrays exist as ndarrays
        even if no elements are released.  So this will always return a valid
        int >= 0.
        """
        return len(self)

    @property
    def data_arrays(self):
        'Returns a dict of the all the data arrays'
        # this is a property in case we want change the internal implementation
        return self._data_arrays

    def keys(self):
        """
        a keys() function so it looks a bit more like a dict
        """
        return self._data_arrays.keys()


class SpillContainer(AddLogger, SpillContainerData):
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
        self.spills = OrderedCollection(dtype=gnome.spills.spill.Spill)
        self.spills.register_callback(self._spills_changed,
                                      ('add', 'replace', 'remove'))
        self.rewind()

    def _reset_arrays(self):
        '''
        reset _array_types dict so it contains default keys/values
        '''

        # copy, cause we don't want to change the defaults!
        self._array_types = {}
        self._data_arrays = {}


    def _reset__substances_spills(self):
        ## Most of this not needed
        '''
        reset internal attributes to None and empty list []:

        1. _substances_spills: data structure to contain spills per substance
        2. _oil_comp_array_len: max number of psuedocomponents - relevant if
           more than one substance is used.
        3. _fate_data_list: list of FateDataView() objects. One object per
           substance if substance is not None

        '''
        # Initialize following either the first time it is used or in
        # prepare_for_model_run() -- it could change with each new spill
        self.substance = None

        # self._substances_spills = None
        self._oil_comp_array_len = 1

    def _reset__fate_data_view(self):
        # define the fate view of the data if 'fate_status' is in data arrays
        # 'fate_status' is included if weathering is on
        self._fate_data_view = FateDataView()

    def reset_fate_dataview(self):
        '''
        reset data arrays for each fate_dataviewer. Each substance that is not
        None has a fate_dataviewer object.
        '''
        self._fate_data_view.reset()
        # for viewer in self._fate_data_list:
        #     viewer.reset()

    def _set_substancespills(self):
        '''
        _substances could change when spills are added/deleted
        using _spills_changed callback to reset self._substance_spills to None

        This checks to make sure that the substance s set correctly and that
        there is not more than one substance

        All spills that are 'on' are included. A spill that is off isn't really
        being modeled so ignore it.

        .. note::
            Should not be called in middle of run. prepare_for_model_run()
            will invoke this if self._substance_spills is None. This is another
            view of the data - it doesn't contain any state that needs to be
            persisted.
        '''
        substance = False #if self.substance is None else self.substance
        for spill in self.spills:
            if not spill.on:
                continue
            if substance is False:  #can't use None, as that means non-weathering
                substance = spill.substance
            else:
                if spill.substance != substance:
                    subs = [spill.substance for spill in self.spills if spill.on]
                    raise ValueError("A spill container can only hold one substance at a time\n"
                                     "trying to add :{}\n"
                                     "These are the substances in the on spills:\n"
                                     "{}".format(substance, subs))
                                     
        # set the number of oil components
        # fixme: with only one substance this could be determined elsewhere
        if hasattr(substance, 'num_components'):
            self._oil_comp_array_len = substance.num_components
        else:
            ## fixme -- is this needed for Non-substance?
            self._oil_comp_array_len = 1

        # it will be False if there are no spills
        self.substance = NonWeatheringSubstance() if substance is False else substance

        #     new_subs = spill.substance
        #     if new_subs in subs:
        #         # substance already defined for another spill
        #         ix = subs.index(new_subs)
        #         spills[ix].append(spill)
        #     else:
        #         # new substance not yet defined
        #         subs.append(new_subs)
        #         spills.append([spill])

        #         # also set _oil_comp_array_len to substance with most
        #         # components? -- *not* being used right now, but make it so
        #         # it works correctly for testing multiple substances
        #         if (hasattr(new_subs, 'num_components') and
        #                 new_subs.num_components > self._oil_comp_array_len):
        #             self._oil_comp_array_len = new_subs.num_components

        # # let's reorder subs so None is in the end:
        # if None in subs:
        #     ix = subs.index(None)
        #     spills.append(spills.pop(ix))
        #     subs.append(subs.pop(ix))

        # s_id = range(len(subs))

        # # # 'data' will be updated when weatherers ask for arrays they need
        # # # define the substances list and the list of spills for each substance
        # # self._substances_spills = substances_spills(substances=subs,
        # #                                             s_id=s_id,
        # #                                             spills=spills)

        # # if len(self.get_substances()) > 1:
        # #     # add an arraytype for substance if more than one substance
        # #     self._array_types.update({'substance': substance})

        # self.logger.info('{0} - number of substances: {1}'.
        #                  format(os.getpid(), len(self.get_substances())))

    def _set_fate_data(self):
        '''
        If the substance is not None, initialize the FateDataView object.
        '''
        # self._fate_data_list = []
        # for s_id, subs in zip(self._substances_spills.s_id,
        #                       self._substances_spills.substances):
        #     if subs is None:
        #         continue

        #     self._fate_data_list.append(FateDataView(s_id))
        self._fate_data_view = FateDataView()

    def _spills_changed(self, *args):
        '''
        call back called on spills add/delete/replace
        Callback simply resets the internal _substance_spills attribute to None
        since the old _substance_spills value could now be invalid.
        '''
        self._set_substancespills()

    def _get_s_id(self, substance):
        '''
        Look in the _substances_spills data structure of substance and return
        the corresponding s_id
        '''
        try:
            ix = self._substances_spills.substances.index(substance)
        except ValueError:
            'substance is not in list'
            self.logger.debug('{0} - Substance named: {1}, not found in data '
                              'structure'.format(os.getpid(), substance.name))
            return None

        return self._substances_spills.s_id[ix]

    def _get_fatedataview(self):
        '''
        return the FateDataView object
        '''
        # ix = self._get_s_id(substance)

        # if ix is None:
        #     msg = "substance named {0} not found".format(substance.name)
        #     self.logger.info(msg)
        #     return

        # # check
        # view = self._fate_data_list[ix]
        # if view.substance_id != ix:
        #     msg = "substance_id did not match as expected. Check!"
        #     raise ValueError(msg)

        return self._fate_data_view

    def _array_name(self, at):
        '''
        given an array type, return the name of the array. This can be string,
        in which case, it is the name of the array so return it. If its not
        a string, then return the at.name attribute.
        '''
        if isinstance(at, str):
            return at
        else:
            return at.name

    def _append_data_arrays(self, num_released):
        """
        initialize data arrays once spill has spawned particles
        Data arrays are set to their initial_values

        :param int num_released: number of particles released

        """
        for name, atype in self._array_types.items():
            # initialize all arrays even if 0 length
            if atype.shape is None:
                # assume array type is for weather data, provide it the shape
                # per the number of components used to model the oil
                # currently, we only have one type of oil, so all spills will
                # model same number of oil_components
                # fixme: is this getting initilazed for non-weathing oil?!?
                #        and is atype.shape is None mean it's neccesarily oil components??
                #        couldn't his be initialized by the spill, which would know how many
                #        components it needs??
                a_append = atype.initialize(num_released,
                                            shape=(self._oil_comp_array_len,),
                                            initial_value=tuple([0] * self._oil_comp_array_len))
            else:
                a_append = atype.initialize(num_released)
            self._data_arrays[name] = np.r_[self._data_arrays[name], a_append]

    # def _set_substance_array(self, subs_idx, num_rel_by_substance):
    #     '''
    #     -. update 'substance' array if more than one substance present. The
    #     value of array is the index of 'substance' in _substances_spills
    #     data structure
    #     '''
    #     if 'substance' in self:
    #         if num_rel_by_substance > 0:
    #             self['substance'][-num_rel_by_substance:] = subs_idx

    def substancefatedata(self,
                          substance,
                          array_types,
                          fate='surface_weather'):
        '''
        Only one substance now!
        todo: fix this so it works for type of fate requested
        return the data for specified substance
        data must contain array names specified in 'array_types'
        '''
        view = self._get_fatedataview()
        return view.get_data(self, array_types, fate)

    def iterspillsbysubstance(self):
        '''
        iterate through the substances spills datastructure and return the
        spills associated with each substance. This is used by release_elements
        DataStructure contains all spills. If some spills contain None for
        substance, these will be returned
        '''
        # fixme -- totally unneccesary??
        # if self._substances_spills is None:
        #     self._set_substancespills()
        return self.spills

    def itersubstancedata(self, array_types, fate_status='surface_weather'):
        '''
        There is only one substance allowed per SpillContainer, so this is
        returns the data cooresponding to the fate_status.

        This is only here to preserve compatiblity

        returns (substance, substance_data)

        This is used by weatherers - if a substance is None, StopIteration is raised

        :param array_types: iterable containing array that should be in the
            data. This could be a set of strings corresponding with array names
            or ArrayType objects which have a name attribute
        :param select='select': a string stating the type of data to be
            returned. Default if 'surface', so all elements with
            status_codes==oil_status.in_water and z == 0 in positions array
        :returns: (substance, substance_data) for each iteration
            substance: substance object
            substance_data: dict of numpy arrays associated with substance with
            elements in_water and on surface if select == 'surface' or
            subsurface if select == 'subsurface'
        '''
        if self.substance is None:
            return []
        else:
            # data = self.data_arrays
            data = self._fate_data_view.get_data(self, array_types, fate_status)
            return [(self.substance, data)]

        # if self._substances_spills is None:
        #     self._set_substancespills()

        # return zip(self.get_substances(complete=False),
        #            [view.get_data(self, array_types, fate) for view in
        #             self._fate_data_list])

    def update_from_fatedataview(self,
                                 # substance=None,
                                 fate_status='surface_weather'):
        '''
        let's only update the arrays that were changed
        only update if a copy of 'data' exists.
        '''
        self._fate_data_view.update_sc(self, fate_status)
        # if substance is not None:
        #     view = self._get_fatedataview(substance)
        #     view.update_sc(self, fate)

        # else:
        #     # do for all substances
        #     for view in self._fate_data_list:
        #         view.update_sc(self, fate)

    def get_substances(self, complete=True):
        ##fixme: remove this method??
        """
        only one substance...
        """
        if self.substance is None:
            return [None] if complete else []
        else:
            return [self.substance]

        # '''
        # return substances stored in _substances_spills structure.
        # Include None if complete is True. Default is complete=True.
        # '''
        # if self._substances_spills is None:
        #     self._set_substancespills()

        # if complete:
        #     return self._substances_spills.substances
        # else:
        #     return filter(None, self._substances_spills.substances)

    @property
    def total_mass(self):
        '''
        return total mass spilled in 'kg'
        '''
        mass = 0
        for spill in self.spills:
            if spill.get_mass() is not None:
                mass += spill.get_mass()

        if mass == 0:
            return None
        else:
            return mass

    @property
    def substances(self):
        '''
        Returns list of substances for weathering - not including None since
        that is non-weathering.
        Currently, only one weathering substance is supported
        '''
        return self.get_substances(complete=False)

    @property
    def array_types(self):
        """
        user can modify ArrayType initial_value in middle of run. Changing
        the shape should throw an error. Change the dtype at your own risk.
        This returns a new dict so user cannot add/delete an ArrayType in
        middle of run. Use prepare_for_model_run() to do add an ArrayType.
        """
        return self._array_types

    def rewind(self):
        """
        In the rewind operation, we:
        - rewind all the spills
        - restore _array_types to contain only defaults
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
        # may have changed so let's get back to default _array_types
        self._reset_arrays()
        self._reset__substances_spills()
        self._reset__fate_data_view()
        self._set_substancespills()
        self.mass_balance = {}  # reset to empty dict

    def get_spill_mask(self, spill):
        return self['spill_num'] == self.spills.index(spill)

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

    def prepare_for_model_run(self, array_types=None, time_step=300):
        """
        called when setting up the model prior to 1st time step
        This is considered 0th timestep by model

        Make current_time optional since SpillContainer doesn't require it
        especially for 0th step; however, the model needs to set it because
        it will write_output() after each step. The data_arrays along with
        the current_time_stamp must be set in order to write_output()

        :param model_start_time: model_start_time to initialize
            current_time_stamp. This is the time_stamp associated with 0-th
            step so initial conditions for data arrays
        :param array_types: a set of additional names and/or array_types to
            append to standard array_types attribute. Set can contain only
            strings or a tuple with (string, ArrayType). See Note below.

        .. note:: set can contains strings or tuples. If set contains only
            strings, say: {'mass', 'windages'},
            then SpillContainer looks for corresponding ArrayType object
            defined in gnome.array_types for 'mass' and 'windages'.
            If set contains a tuple, say: {('mass', gnome.array_types.mass)},
            then SpillContainer uses the ArrayType defined in the tuple.

        .. note:: The SpillContainer iterates through each of the item in
            array_types and checks to see if there is an associated initializer
            in any Spill. If corresponding initializer is found, it gets the
            array_types from initializer and appends them to its own list. This
            was added for the case where 'droplet_diameter' array is
            defined/used by initializer (InitRiseVelFromDropletSizeFromDist)
            and we would like to see it in output, but no Mover/Weatherer needs
            it.
        """
        # Question - should we purge any new arrays that were added in previous
        # call to prepare_for_model_run()?
        # No! If user made modifications to _array_types before running model,
        # let's keep those. A rewind will reset data_arrays.
        if array_types is None:
            array_types = {}

        #self._append_initializer_array_types(array_types)
        for s in self.spills:
            s.prepare_for_model_run(time_step)
        ats = default_array_types.copy()
        ats.update(array_types)
        self._array_types = ats

        # if self._substances_spills is None:
        #     self._set_substancespills()

        # also create fate_dataview if 'fate_status' is part of arrays
        if 'fate_status' in self.array_types:
            self._set_fate_data()

        # 'substance' data_array may have been added so initialize after
        # _set_substancespills() is invoked
        self._set_substancespills()
        self.initialize_data_arrays()

        # todo: maybe better to let map do this, but it does not have a
        # prepare_for_model_run() yet so can't do it there
        # need 'amount_released' here as well
        self.mass_balance['beached'] = 0.0
        self.mass_balance['off_maps'] = 0.0

    def initialize_data_arrays(self):
        """
        initialize_data_arrays() is called without input data during rewind
        and prepare_for_model_run to define all data arrays.
        At this time the arrays are empty.
        """
        for name, atype in self._array_types.items():
            # Initialize data_arrays with 0 elements
            # fixme: is every array type with None shape neccesarily
            #        oil components??
            #        but it is more than just mass_components
            #        maybe some other flag??
            if atype.shape is None:
                num_comp = self._oil_comp_array_len
                self._data_arrays[name] = atype.initialize_null(shape=(num_comp,))
            else:
                self._data_arrays[name] = atype.initialize_null()

    def _get_fate_mask(self, fate):
        '''
        get fate_status mask over SC - only include LEs with 'mass' > 0.0
        '''
        if fate == 'all':
            # look at all fate data
            w_mask = np.asarray([True] * len(self))
        else:
            w_mask = (self['fate_status'] & getattr(bt_fate, fate) ==
                      getattr(bt_fate, fate))

        w_mask = np.logical_and(w_mask, self['mass'] > 0.0)
        return w_mask

    def release_elements(self, start_time, end_time, environment=None):
        """
        :param start_time: -- beginning of the release
        :param end_time: -- end of the release.

        This calls release_elements on all of the contained spills, and adds
        the elements to the data arrays

        :returns: total number of particles released
        """
        total_rel = 0
        # substance index - used label elements from same substance
        # used internally only by SpillContainer - could be a strided array.
        # Simpler to define it only in SpillContainer as opposed to ArrayTypes
        # 'substance': ((), np.uint8, 0)
        for spill in self.spills:
            # only want to include the spills that are turned on.
            if not spill.on:
                continue

            num_rel = spill.release_elements(self, start_time, end_time, environment=environment)
            if num_rel > 0:
                # update 'spill_num' ArrayType's initial_value so it
                # corresponds with spill number for this set of released
                # particles - just another way to set value of spill_num
                # correctly
                self._array_types['spill_num'].initial_value = \
                    self.spills.index(spill)

                if len(self['spill_num']) > 0:
                    # unique identifier for each new element released
                    # this adjusts the _array_types initial_value since the
                    # initialize function just calls:
                    #  range(initial_value, num_released + initial_value)
                    self._array_types['id'].initial_value = \
                        self['id'][-1] + 1
                else:
                    # always reset value of first particle released to 0!
                    # The array_types are shared globally. To initialize
                    # uncertain spills correctly, reset this to 0.
                    # To be safe, always reset to 0 when no
                    # particles are released
                    self._array_types['id'].initial_value = 0

                # append to data arrays - number of oil components is
                # currently the same for all spills
                total_rel += num_rel

        # reset fate_dataview at each step - do it after release elements
        # fixme: now that release_elements is called twice -- maybe not
        #        the place to do it?
        self.reset_fate_dataview()
        return total_rel

    def split_element(self, ix, num, l_frac=None):
        '''
        split an element into specified number.
        For data, like mass, that gets divided, l_frac can be optionally
        provided. l_frac is a list containing fraction of component's value
        given to each new element. len(l_frac) must be equal to num and
        sum(l_frac) == 1.0

        :param ix: id of element to be split - before splitting each element
            has a unique 'id' defined in 'id' data array
        :type ix: int
        :param num: split ix into 'num' number of elements
        :type num: int
        :param l_frac: list containing fractions that sum to 1.0 with
            len(l_frac) == num
        :type l_frac: list or tuple or numpy array
        '''
        # split the first location where 'id' matches
        try:
            idx = np.where(self['id'] == ix)[0][0]
        except IndexError:
            msg = "no element with id = {0} found".format(ix)
            self.logger.warning(msg)
            raise

        for name, at in self.array_types.items():
            data = self[name]
            split_elems = at.split_element(num, self[name][idx], l_frac)
            data = np.insert(data, idx, split_elems[:-1], 0)
            data[idx + len(split_elems) - 1] = split_elems[-1]
            self._data_arrays[name] = data

        # update fate_dataview which contains this LE
        # for now we only have one type of substance
        self._fate_data_view._reset_fatedata(self, ix)

    def model_step_is_done(self):
        '''
        Called at the end of a time step
        Need to remove particles marked as to_be_removed...
        '''
        if len(self._data_arrays) == 0:
            return  # nothing to do - arrays are not yet defined.

        # LEs are marked as to_be_removed
        # C++ might care about this so leave as is
        to_be_removed = np.where(self['status_codes'] ==
                                 oil_status.to_be_removed)[0]

        if len(to_be_removed) > 0:
            for key in self._array_types:
                self._data_arrays[key] = np.delete(self[key], to_be_removed,
                                                   axis=0)
            self._fate_data_view.reset()

    def __str__(self):
        return ('gnome.spill_container.SpillContainer\n'
                'spill LE attributes: {0}'
                .format(sorted(self._data_arrays.keys())))

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
        'Initialize object with the spill_containers passed in'
        if sc.uncertain:
            raise ValueError('sc is an uncertain SpillContainer')

        self._spill_container = sc

        if u_sc is None:
            self._uncertain = False
        else:
            self._uncertain = True

            if not u_sc.uncertain:
                raise ValueError('u_sc is not an uncertain SpillContainer')

            self._u_spill_container = u_sc

    def __repr__(self):
        return ('{0.__class__.__name__},\n'
                '  uncertain={0.uncertain}\n '.format(self))

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

        NOTE: cache code counts on the uncertain SpillContainer being last
        """
        if self.uncertain:
            return (self._spill_container, self._u_spill_container)
        else:
            return (self._spill_container,)

    @property
    def LE_data(self):
        data = list(self._spill_container._data_arrays.keys())
        data.append('current_time_stamp')
        if self._spill_container.mass_balance:
            'only add if it is not an empty dict'
            data.append('mass_balance')

        return data

    def LE(self, prop_name, uncertain=False):
        if uncertain:
            sc = self._u_spill_container
        else:
            sc = self._spill_container

        if prop_name == 'current_time_stamp':
            return sc.current_time_stamp
        elif prop_name == 'mass_balance':
            return sc.mass_balance

        return sc[prop_name]

    def __eq__(self, other):
        'Compare equality of two SpillContainerPairData objects'
        if type(self) != type(other):
            return False

        if self.uncertain != other.uncertain:
            return False

        for sc in zip(self.items(), other.items()):
            if sc[0] != sc[1]:
                return False

        return True

    def __ne__(self, other):
        return not (self == other)


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
        'rewind spills in spill_container'
        self._spill_container.rewind()

        if self.uncertain:
            self._u_spill_container.rewind()
            if self._spill_container.spills != self._u_spill_container.spills:
                self._u_spill_container = \
                    self._spill_container.uncertain_copy()

    def __repr__(self):
        'unambiguous repr'
        return ('{0.__class__.__name__},\n'
                '  uncertain={0.uncertain}\n'
                '  Spills: {1}'.format(self, self._spill_container.spills))

    @property
    def uncertain(self):
        return self._uncertain

    @uncertain.setter
    def uncertain(self, value):
        if type(value) is not bool:
            raise TypeError("uncertain property must be a bool (True/False)")

        if self._uncertain is True and value is False:
            self._uncertain = value
            del self._u_spill_container  # delete if it exists
        elif self._uncertain is False and value is True:
            self._uncertain = value
            self._u_spill_container = self._spill_container.uncertain_copy()

    def _add_spill_pair(self, pair_tuple):
        'add both certain and uncertain spills given as a pair'
        if self.uncertain and len(pair_tuple) != 2:
            raise ValueError('You can only add a tuple containing a '
                             'certain/uncertain spill pair '
                             '(spill, uncertain_spill)')
        if not self.uncertain and len(pair_tuple) != 1:
            raise ValueError('Uncertainty is off. Tuple must only '
                             'contain (certain_spill,)')

        self._spill_container.spills += pair_tuple[0]
        if self.uncertain:
            self._u_spill_container.spills += pair_tuple[1]

    def _add_item(self, item):
        'could be a spill pair or a forecast spill - add appropriately'
        if isinstance(item, tuple):
            # add both certain and uncertain pair
            self._add_spill_pair(item)
        else:
            self._spill_container.spills += item
            if self.uncertain:
                self._u_spill_container.spills += item.uncertain_copy()

    def add(self, spills):
        """
        Add spill to spill_container and make copy in u_spill_container
        if uncertainty is on

        Note: Method can take either a list, tuple, or list of tuples
              with following assumptions:

        1. spills = Spill()    # A spill object, if uncertainty is on, make a
        copy for uncertain_spill_container.

        2. spills = [s0, s1, ..,]    # List of forecast spills. if uncertain,
        make a copy of each and add to uncertain_spill_container

        3. spills = (s0, uncertain_s0)    # tuple of length two. Assume first
        one is forecast spill and second one is the uncertain copy. Used
        when restoring from save file

        4. spills = [(s0, uncertain_s0), ..]    # list of tuples of length two.
        Added for completeness.
        """
        if isinstance(spills, list):
            for item in spills:
                self._add_item(item)
        else:
            # only adding one item, either a spill_pair or a forecast spill
            self._add_item(spills)

    def append(self, spill):
        self.add(spill)

    def remove(self, ident):
        '''
        remove object from spill_container.spills and the corresponding
        uncertainty spill as well
        '''
        if self.uncertain:
            'ident could be index or object so handle both'
            idx = self._spill_container.spills.index(
                self._spill_container.spills[ident])
            del self._u_spill_container.spills[idx]

        del self._spill_container.spills[ident]

    def __getitem__(self, ident):
        'only return the certain spill'
        spill = self._spill_container.spills[ident]
        return spill

    def __setitem__(self, ident, new_spill):
        self._spill_container.spills.replace(ident, new_spill)
        if self.uncertain:
            ix = self.index(new_spill)
            self._u_spill_container.spills[ix] = new_spill.uncertain_copy()

    def __delitem__(self, ident):
        self.remove(ident)

    def __iadd__(self, rop):
        self.add(rop)
        return self

    def __iter__(self):
        'iterates over the spills defined in spill_container'
        for sp in self._spill_container.spills:
            yield self.__getitem__(sp.id)

    def __len__(self):
        '''
        It refers to the total number of spills that have been added
        The uncertain and certain spill containers will contain the same number
        of spills return the length of spill_container.spills
        '''
        return len(self._spill_container.spills)

    def __contains__(self, ident):
        '''
        looks to see if ident which is the id of a spill belongs in the
        _spill_container.spills OrderedCollection
        '''
        return ident in self._spill_container.spills

    def to_dict(self):
        """
        takes the instance of SpillContainerPair class and outputs a dict with:
            'spills': call to_dict() on spills ordered collection
            stored in certain spill container

        if uncertain, then also return:
            'uncertain_spills': call to_dict() on spills ordered collection
            stored in uncertain spill container

        The input param json_ is not used. It is there to keep the same
        interface for all to_dict() functions
        """
        dict_ = {'spills':
                 self._spill_container.spills.to_dict()}
        if self.uncertain:
            dict_.update({'uncertain_spills':
                          self._u_spill_container.spills.to_dict()})
        return dict_

    def update_from_dict(self, dict_):
        '''
        takes a dict {'spills': [list of spill objects]}, checks them against
        the forecast spills contained in _spill_container.spills and updates
        if they are different

        It also creates a copy of the different spill and replaces the
        corresponding spill in _u_spill_container

        This is primarily intended for the webapp so the dict_ will only
        contain a list of forecast spills
        '''
        l_spills = dict_['spills']
        updated = False
        if len(l_spills) != len(self):
            updated = True

        if list(self._spill_container.spills.values()) != l_spills:
            updated = True

        if updated:
            self.clear()
            if l_spills:
                self += l_spills
        return updated

    def spill_by_index(self, index, uncertain=False):
        '''return either the forecast spill or the uncertain spill at
        specified index'''
        if uncertain:
            return self._u_spill_container.spills[index]
        else:
            # __getitem__ should give correct result
            return self[index]

    def index(self, spill):
        '''
        Look for spill in forecast SpillContainer or uncertain SpillContainer
        and return the index of ordered collection where spill is found
        '''
        try:
            return self._spill_container.spills.index(spill)
        except Exception:
            return self._u_spill_container.spills.index(spill)

    @property
    def num_released(self):
        'elements released by (forecast, uncertain) spills'
        if self.uncertain:
            return (self._spill_container.num_released,
                    self._u_spill_container.num_released)
        else:
            return (self._spill_container.num_released,)

    def clear(self):
        'clear all spills from container pairs'
        self._spill_container.spills.clear()
        if self.uncertain:
            self._u_spill_container.spills.clear()
