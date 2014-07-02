"""
module contains objects that contain weather related data. For example,
the Wind object defines the Wind conditions for the spill
"""
import copy

from gnome.utilities import serializable


class Environment(object):
    """
    A base class for all classes in environment module

    This is primarily to define a dtype such that the OrderedCollection
    defined in the Model object requires it.
    """
    _state = copy.deepcopy(serializable.Serializable._state)

    def __init__(self, name=None):
        '''
        base class for environment objects

        :param name=None:
        '''
        if name:
            self.name = name 