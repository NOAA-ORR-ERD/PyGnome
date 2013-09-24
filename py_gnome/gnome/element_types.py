'''
Module contains array types that a mover can contain based on the data
it needs for the released elements.

The different types of arrays are stored in a frozenset so they are read only

The users of this data can convert it to a dict or any other data type that is
useful
'''

from gnome.basic_types import (
    world_point_type,
    windage_type,
    status_code_type,
    oil_status,
    id_type)
import numpy as np


class ArrayType(object):

    """
    Object used to capture attributes of numpy data array for elements

    An ArrayType specifies how data arrays associated with elements
    are defined.
    """

    def __init__(
        self,
        shape,
        dtype,
        initial_value=0,
        ):
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

    def initialize(self, num_elements, spill):
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
        """" Equality of two ArrayType objects """

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
        """
        Compare inequality (!=) of two objects
        """

        if self == other:
            return False
        else:
            return True

#==============================================================================
# all_spills = frozenset([('positions', ArrayType((3, ),
#                        basic_types.world_point_type)), ('mass',
#                        ArrayType((), np.float64))])
# 
# all_spill_containers = frozenset([('next_positions', ArrayType((3, ),
#                                  basic_types.world_point_type)),
#                                  ('last_water_positions', ArrayType((3,
#                                  ), basic_types.world_point_type)),
#                                  ('status_codes', ArrayType((),
#                                  basic_types.status_code_type,
#                                  basic_types.oil_status.in_water)),
#                                  ('spill_num', ArrayType((),
#                                  basic_types.id_type, -1))])
#
# windage = frozenset([('windages', ArrayType((),
#                     basic_types.windage_type))])
#==============================================================================

Mover = {'positions': ArrayType((3,), world_point_type),
         'mass': ArrayType((), np.float64),
         'next_positions': ArrayType((3,), world_point_type),
         'last_water_positions': ArrayType((3,), world_point_type),
         'status_codes': ArrayType((), status_code_type,
                                   oil_status.in_water),
         'spill_num': ArrayType((), id_type, -1)
        }

WindMover = {'windages': ArrayType((), windage_type)}

# droplet_size, rise_vel arrays used by spills

#droplet_size = frozenset([('droplet_size', ArrayType((), np.float64))])
#rise_vel = frozenset([('rise_vel', ArrayType((), np.float64))])
RiseVelocityMover = {'rise_vel': ArrayType((), np.float64)}

## TODO: Find out if this is still required?
# subsurface = {'water_currents':ArrayType( (3,), basic_types.water_current_type)}
