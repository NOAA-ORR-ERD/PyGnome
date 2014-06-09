#!/usr/bin/env python
'''
Module contains array types that a mover may need based on the data
movers needs

** NOTE: **
    These are global declarations

    For instance: If the WindMover that uses array_types.WindMover updates
    the properties of 'windages' ArrayType, it will change it universally.

    The user/mover should not need to change dtype or shape internally. If
    these need to change, it should be done here.

    The initial_value can be changed by movers since that's only used when
    elements are released. For most arrays, that is currently 0.

    As a convention, when a dict defines these array_types, best to use the
    name of the array_type as the 'key'. When other modules, primarily
    element_type, look for numpy array in data_arrays with associated
    array_types, it will assume the 'key' is the name of the array_types.

'''

import sys
import inspect

from gnome.basic_types import (world_point_type,
                               windage_type,
                               status_code_type,
                               oil_status,
                               id_type)

import numpy
np = numpy


class ArrayType(object):
    """
    Object used to capture attributes of numpy data array for elements

    An ArrayType specifies how data arrays associated with elements
    are defined.
    """
    def __init__(self, shape, dtype, initial_value=0):
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

    def initialize_null(self):
        """
        initialize array with 0 elements. Used so SpillContainer can
        initializes all arrays with 0 elements. Used when the model is rewound.
        The purpose is to show all data_arrays even if model is not yet running
        or no particles have been released
        """
        return self.initialize(0)

    def initialize(self, num_elements):
        """
        Initialize a numpy array with the dtype and shape specified. The length
        of the array is given by num_elements and spill is given as input if
        the initialize function needs information about the spill to initialize

        :param num_elements:
        :param spill:
        """
        arr = np.zeros((num_elements,) + self.shape, dtype=self.dtype)
        if self.initial_value != 0:
            arr[:] = self.initial_value
        return arr

    def __eq__(self, other):
        if not isinstance(other, self.__class__):
            return False

        if len(self.__dict__) != len(other.__dict__):
            return False

        for (key, val) in self.__dict__.iteritems():
            if key not in other.__dict__:
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
    def initialize(self, num_elements):
        array = np.arange(self.initial_value,
                          num_elements + self.initial_value, dtype=self.dtype)
        return array


# SpillContainer manipulates initial_value property to initialize 'spill_num'
# and 'element_id' properly. Referencing global ArrayType objects for this
# means the initial values may not get reset. Need a function to reset
# to default values
_default_values = {
     'positions': ((3,), world_point_type, (0., 0., 0.)),
     'next_positions': ((3,), world_point_type, (0., 0., 0.)),
     'last_water_positions': ((3,), world_point_type, (0., 0., 0.)),
     'status_codes': ((), status_code_type, oil_status.in_water),
     'spill_num': ((), id_type, 0),
     'id': ((), np.uint32, 0, IdArrayType),
     'mass': ((), np.float64, 0),
     'windages': ((), windage_type, 0),
     'windage_range': ((2,), np.float64, (0., 0.)),
     'windage_persist': ((), np.int, 0),
     'rise_vel': ((), np.float64, 0.),
     'droplet_diameter': ((), np.float64, 0.),
     'age': ((), np.int32, 0),
     'mass_components': ((5,), np.float64, (1., 0., 0., 0., 0.)),
     'half_lives': ((5,), np.float64,
                    (np.inf, np.inf, np.inf, np.inf, np.inf)),
      }


# dynamically create the ArrayType objects in this module from _default_values
# dict
for key, val in _default_values.iteritems():
    if len(val) > 3:
        vars()[key] = val[3](shape=_default_values[key][0],
                             dtype=_default_values[key][1],
                             initial_value=_default_values[key][2])
    else:
        vars()[key] = ArrayType(shape=_default_values[key][0],
                                dtype=_default_values[key][1],
                                initial_value=_default_values[key][2])


# use reflection to:
#    define a function to reset all ArrayTypes to defaults
_to_reset = inspect.getmembers(sys.modules[__name__],
                               predicate=lambda members:
                               (False, True)[isinstance(members, ArrayType)])
_at_names = [item[0] for item in _to_reset]


def reset_to_defaults(names=_at_names):
    for item in _to_reset:
        if item[0] in names:
            obj = eval(item[0])
            obj.shape = _default_values[item[0]][0]
            obj.dtype = _default_values[item[0]][1]
            obj.initial_value = _default_values[item[0]][2]
