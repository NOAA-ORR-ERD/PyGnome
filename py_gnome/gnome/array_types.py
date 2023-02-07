#!/usr/bin/env python
'''
Module contains array types that a mover may need based on the data
movers needs

** NOTE: **
    These are global declarations

    For instance: If the PointWindMover that uses array_types.PointWindMover updates
    the properties of 'windages' ArrayType, it will change it universally.

    The user/mover should not need to change dtype or shape internally. If
    these need to change, it should be done here.

    The initial_value can be changed by movers since that's only used when
    elements are released. For most arrays, that is currently 0.

    As a convention, when a dict defines these array_types, best to use the
    name of the array_type as the 'key'. When other modules, primarily
    element_type, look for numpy array in data_arrays with associated
    array_types, it will assume the 'key' is the name of the array_types.

    todo: For array_types like mass_components and other array_types used by
    weathering objects, the shape is set at runtime. Since this changes
    depending on each type of spill that is modeled, need to rethink this
    global definition for these array_types. Currently, these ArrayTypes are
    defined with 'shape'=None. Add optional shape argument to
    ArrayTypes().initialize() to handle the case where shape attribute is None

'''





import sys

import numpy as np

from gnome.basic_types import (world_point_type,
                               windage_type,
                               status_code_type,
                               oil_status,
                               id_type,
                               fate)
from gnome import AddLogger


class ArrayType(AddLogger):
    """
    Object used to capture attributes of numpy data array for elements

    An ArrayType specifies how data arrays associated with elements
    are defined.
    """
    def __init__(self, shape, dtype, name, initial_value=0):
        """
        constructor for ArrayType

        :param shape: shape of the numpy array
        :type shape: tuple of integers
        :param dtype: numpy datatype contained in array
        :type dtype: numpy dtype
        """
        self.shape = shape
        self.dtype = dtype
        self.initial_value = initial_value
        self.name = name

    def __repr__(self):
        return "ArrayType(name={0}, initial_value={1}, shape={2}, dtype={3}".format(self.name, self.initial_value, self.shape, self.dtype)

    def initialize_null(self, shape=None):
        """
        initialize array with 0 elements. Used so SpillContainer can
        initializes all arrays with 0 elements. Used when the model is rewound.
        The purpose is to show all data_arrays even if model is not yet running
        or no particles have been released
        """
        return self.initialize(0, shape)

    def initialize(self, num_elements, shape=None, initial_value=None):
        """
        Initialize a numpy array with the dtype and shape specified. The length
        of the array is given by num_elements and spill is given as input if
        the initialize function needs information about the spill to initialize

        :param num_elements: number of elements so size of array to initialize

        Optional parameter

        :param shape=None: If this is None then use self.shape to determine
            size of array to create, else use this parameter. This is primarily
            used for ArrayTypes where either object's shape attribute is None
            or we want to override the object's predefined 'shape' during
            initialization
        """
        if shape is None:
            shape = self.shape

        if initial_value is None:
            initial_value = self.initial_value

        arr = np.zeros((num_elements,) + shape, dtype=self.dtype)
        arr[:] = initial_value

        return arr

    def _num_gt_2(self, num):
        if num < 2:
            msg = "'num' to split into must be at least 2"
            self.logger.error(msg)
            raise ValueError(msg)

    def split_element(self, num, value, *args):
        '''
        define how an LE gets split for specified ArrayType

        :param num: number of elements that current value should get split into
        :type num: int
        :param value: the current value that is replicated
        :type value: this must have shape and dtype equal to self.shape
            and self.dtype
        :param *args: accept more arguments as derived class may divide LE on
            split and in this case, user can specify a list of fractions for
            this division.
        '''
        self._num_gt_2(num)

        shape = value.shape if self.shape is None else self.shape
        return self.initialize(num, shape, value)

    def __eq__(self, other):
        if not isinstance(other, self.__class__):
            return False

        if len(self.__dict__) != len(other.__dict__):
            return False

        for (key, val) in self.__dict__.items():
            if key not in other.__dict__:
                return False
            elif key == 'initial_value':
                if np.any(val != other.__dict__[key]):
                    return False
            elif val != other.__dict__[key]:
                return False

        # everything passed, then they must be equal
        return True

    def __ne__(self, other):
        return not self == other


class IdArrayType(ArrayType):
    """
    The 'id' array assigns a unique int for every particle released.
    """
    def initialize(self, num_elements, *args):
        '''
        overrides base initialize functionality to output a range of values

            self.initial_value to num_elements + self.initial_value

        This is only used for 'id' of particle.
        shape attribute and initial_value are ignored
        since you always get an array of shape (num_elements,)
        Define *args to keep method signature the same
        '''
        array = np.arange(self.initial_value,
                          num_elements + self.initial_value, dtype=self.dtype)
        return array

    def split_element(self, num, value, *args):
        '''
        split elements into num and assign 'value' to all of them
        '''
        self._num_gt_2(num)

        arr = np.zeros((num,) + self.shape, dtype=self.dtype)
        arr[:] = value

        return arr


class ArrayTypeDivideOnSplit(ArrayType):
    def split_element(self, num, value, l_frac=None):
        '''
        define how an LE gets split for specified ArrayType. l_frac if given
        should sum to 1.0

        :param num: number of elements that current value should get split into
        :type num: int
        :param value: the current value that is replicated
        :type value: this must have shape and dtype equal to self.shape
            and self.dtype
        :param l_frac: user can specify a list of fractions for
            this division - if None, then evenly divide 'value' into 'num'.
            sum(l_frac) = 1.0
        '''
        self._num_gt_2(num)

        if l_frac is not None:
            if len(l_frac) != num:
                msg = "in split_element() len(l_frac) must equal 'num'"
                self.logger.error(msg)
                raise ValueError(msg)

            l_frac = np.asarray(l_frac)
            if not np.allclose(l_frac.sum(), 1.0):
                msg = "sum 'l_frac' must be 1.0"
                self.logger.error(msg)
                raise ValueError(msg)

        shape = value.shape if self.shape is None else self.shape
        split = self.initialize(num, shape, value)

        if l_frac is None:
            return split[:]/num
        else:
            if len(split.shape) > 1:
                # 2D array so reshape l_frac
                return split * l_frac.reshape(len(l_frac), -1)
            else:
                return split * l_frac


# SpillContainer manipulates initial_value property to initialize 'spill_num'
# and 'element_id' properly. Referencing global ArrayType objects for this
# means the initial values may not get reset. Need a function to reset
# to default values
_default_values = {'positions': ((3,), world_point_type, 'positions',
                                 (0., 0., 0.)),
                   'next_positions': ((3,), world_point_type, 'next_positions',
                                      (0., 0., 0.)),
                   'last_water_positions': ((3,), world_point_type,
                                            'last_water_positions',
                                            (0., 0., 0.)),
                   'status_codes': ((), status_code_type, 'status_codes',
                                    oil_status.in_water),
                   'spill_num': ((), id_type, 'spill_num', 0),
                   'id': ((), np.uint32, 'id', 0, IdArrayType),
                   'windages': ((), windage_type, 'windages', 0),
                   'windage_range': ((2,), np.float64, 'windage_range',
                                     (0., 0.)),
                   'windage_persist': ((), np.int32, 'windage_persist', 0),
                   'rise_vel': ((), np.float64, 'rise_vel', 0.),
                   'droplet_diameter': ((), np.float64, 'droplet_diameter',
                                        0.),
                   'age': ((), np.int32, 'age', 0),

                   # WEATHERING DATA
                   # following used to compute spreading (LE thickness)
                   # bulk_init_volume initial volume of blob of oil - the sum
                   # of all LEs released together is the volume of the blob.
                   # It is evenly divided to number of LEs
                   'vol_frac_le_st': ((), np.float64, 'vol_frac_le_st', 0),
                   'max_area_le': ((), np.float64, 'max_area_le', 0),
                   'release_rate': ((), np.float64, 'release_rate', np.nan),
                   'bulk_init_volume': ((), np.float64, 'bulk_init_volume', 0,
                                        ArrayTypeDivideOnSplit),
                   'density': ((), np.float64, 'density', 1000),
                   'oil_density': ((), np.float64, 'oil_density', 0),
                   'evap_decay_constant': (None, np.float64,
                                           'evap_decay_constant', None),

                   # area is frac_coverage * fay_area - it is the area adjusted
                   # by langmuir. Objects should only use 'area' array, but
                   # keep 'fay_area' and 'frac_coverage' for diagnostics
                   'fay_area': ((), np.float64, 'fay_area', 0),
                   'area': ((), np.float64, 'area', 0),
                   'frac_coverage': ((), np.float64, 'frac_coverage', 1.0),

                   # decided not to use bool since netcdf needs a primitive
                   # type. The conversion would need to happen between bool on
                   # write and read in NetCDFOutput - requires more code so
                   # decided to make it a uint8 instead
                   'at_max_area': ((), np.uint8, 'at_max_area', False),

                   'viscosity': ((), np.float64, 'viscosity', 1000000000000.),
                   'oil_viscosity': ((), np.float64, 'oil_viscosity', 0),
                   # fractional water content in emulsion
                   'frac_water': ((), np.float64, 'frac_water', 0),
                   'interfacial_area': ((), np.float64, 'interfacial_area', 0),
                   # internal factor for biodegradation
                   'yield_factor': ((), np.float64, 'yield_factor', 0),
                   # use negative as a not yet set flag
                   'bulltime': ((), np.float64, 'bulltime', -1.),
                   'frac_lost': ((), np.float64, 'frac_lost', 0),
                   'frac_evap': ((), np.float64, 'frac_evap', 0),

                   # substance index - used label elements from same substance
                   # used internally only by SpillContainer *if* more than one
                   # substance
                   'substance': ((), np.uint8, 'substance', 0),
                   'fate_status': ((), np.uint8, 'fate_status',
                                   fate.non_weather),

                   # Following objects will divide value of element when
                   # calling split_element(), use: ArrayTypeDivideOnSplit()
                   'mass': ((), np.float64, 'mass', 0, ArrayTypeDivideOnSplit),
                   'mass_components': (None, np.float64, 'mass_components',
                                       None, ArrayTypeDivideOnSplit),
                   # The initial mass of the element
                   # used to compute frac of mass lost
                   'init_mass': ((), np.float64, 'init_mass', 0,
                                 ArrayTypeDivideOnSplit),
                   'partition_coeff': ((), np.float64, 'partition_coeff', 0),
                   'droplet_avg_size': ((), np.float64, 'droplet_avg_size', 0),
                   'surface_concentration': ((), np.float64, 'surface_concentration', 0),
                   }

def get_array_type(name):
    """
    Returns and instance of an array type appropriate for name, or None if one
    does not exist
    """
    params = _default_values[name]
    # special case for IdArrayType and ArrayTypeDivideOnSplit
    if len(params)>4:
        return params[4](shape=params[0],
                         dtype=params[1],
                         name=params[2],
                         initial_value=params[3])

    else:
        return ArrayType(shape=params[0],
                         dtype=params[1],
                         name=params[2],
                         initial_value=params[3])

gat = get_array_type

# dynamically create the ArrayType objects in this module from _default_values
# dict. Reason for this logic and subsequent functions is so we only have to
# update/modify the _default_values dict above
# for key, val in _default_values.iteritems():
#     if len(val) > 4:
#         vars()[key] = val[4](shape=_default_values[key][0],
#                              dtype=_default_values[key][1],
#                              name=_default_values[key][2],
#                              initial_value=_default_values[key][3])
#     else:
#         vars()[key] = ArrayType(shape=_default_values[key][0],
#                                 dtype=_default_values[key][1],
#                                 name=_default_values[key][2],
#                                 initial_value=_default_values[key][3])
#
#
# # list of names of all ArrayTypes defined in this module
# mod = sys.modules[__name__]


#    define a function to reset all ArrayTypes to defaults
def reset_to_defaults(at):
        try:
            obj = at
            obj.shape = _default_values[at.name][0]
            obj.dtype = _default_values[at.name][1]
            obj.name = _default_values[at.name][2]
            obj.initial_value = _default_values[at.name][3]

        except AttributeError:
            # name is not part of the defaults - ignore it
            pass

# The array types that will always be used in the model.

# fixme: why are viscosity and surface_concentration here? even density?
DEFAULT_ARRAY_TYPES = ['positions',
                       'next_positions',
                       'last_water_positions',
                       'status_codes',
                       'mass',
                       'init_mass',
                       'age',
                       'density',
                       'viscosity',
                       'surface_concentration',
                       'spill_num',
                       'id',
                       'vol_frac_le_st',
                       'max_area_le',
                       'release_rate',
                       'bulk_init_volume',
                       'area',
                       'fay_area',
                       'frac_coverage',
                       ]

default_array_types = {at: gat(at) for at in DEFAULT_ARRAY_TYPES}
