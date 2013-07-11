'''
Module contains types of arrays that a mover can contain based on the data
it needs for the released elements
'''
from gnome import basic_types

class ArrayType(object):#,serializable.Serializable):
    """
    Object used to capture attributes of numpy data array for elements

    An ArrayType specifies how data arrays associated with elements
    are defined.

    Used by :class:`Spill` and :class:`gnome.spill_container.SpillContainer` 
    """
    
    def __init__(self, shape, dtype, initial_value=None):
        """
        constructor for ArrayType
        
        :param shape: shape of the numpy array
        :type shape: tuple of integers
        :param dtype: numpy datatype contained in array
        :type dtype: numpy dtype
        :param initial_value: initialize array to this value
        :type initial_value: numpy array of size: shape(:-1) (ie. the shape of a single element)
        """
        self.shape = shape
        self.dtype = dtype
        self.initial_value = initial_value

    def __eq__(self, other):
        """" Equality of two ArrayType objects """
        if not isinstance(other, self.__class__):
            return False
        
        if len(self.__dict__) != len(other.__dict__):   
            return False
        
        for key,val in self.__dict__.iteritems():
             if key not in other.__dict__:
                 return False
             
             elif val != other.__dict__[key]:
                 return False
             
        # everything passed, then they must be equal
        return True
    
    def __ne__(self,other):
        """ 
        Compare inequality (!=) of two objects
        """
        if self == other:
            return False
        else:
            return True
        
    
class basic(object):
    """
    All released elements must contain these 'basic' arrays.
    """
    #===========================================================================
    # _positions_ = ArrayType( (3,), basic_types.world_point_type)
    # _next_positions_ = ArrayType( (3,), basic_types.world_point_type)
    # _last_water_positions_ = ArrayType( (3,), basic_types.world_point_type)
    # _status_codes_ = ArrayType( (), basic_types.status_code_type,
    #                         basic_types.oil_status.in_water)
    # _spill_num_ = ArrayType( (), basic_types.id_type)
    # 
    # @classmethod
    # def array_types(cls):
    #    return {'positions':cls._positions_,
    #            'next_positions':cls._next_positions_}
    #===========================================================================
    
    @property
    def array_types(self):
        """
        property returns a dict that contains the name of the array for the key and
        the ArrayType object as the value
        contains: position, next_position, last_water_position, status_code, spill_num
        """
        etypes = {'positions':ArrayType( (3,), basic_types.world_point_type),
                 'next_positions': ArrayType( (3,), basic_types.world_point_type),
                 'last_water_positions': ArrayType( (3,), basic_types.world_point_type),
                 'status_codes': ArrayType( (), basic_types.status_code_type,basic_types.oil_status.in_water),
                 'spill_num': ArrayType( (), basic_types.id_type, -1)}
        return etypes
        
class windage(object):
    @property
    def array_types(self):
        """
        property returns a dict that contains the name of the array for the key and
        the ArrayType object as the value
        contains: windage
        """ 
        etypes = {'windages':ArrayType( (), basic_types.windage_type)}
        return etypes
        
class subsurface(object):
    @property
    def array_types(self):
        """
        property returns a dict that contains the name of the array for the key and
        the ArrayType object as the value
        contains: water_current 
        """ 
        etypes = {'water_currents':ArrayType( (3,), basic_types.water_current_type)}
        return etypes