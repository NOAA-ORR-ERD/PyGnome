"""
module contains objects that contain weather related data. For example,
the Wind object defines the Wind conditions for the spill
"""

import copy

from gnome import GnomeId
from gnome.utilities import serializable


class Environment(object):

    """
    A base class for all classes in environment module

    This is primarily to define a dtype such that the OrderedCollection
    defined in the Model object requires it.

    This base class just defines the id property
    """
    _state = copy.deepcopy(serializable.Serializable._state)

    def __init__(self, **kwargs):
        """
        Base class - serves two purposes:
        1) Defines the dtype for all objects that can be added to the Model's
           environment OrderedCollection (Model.environment)
        2) Defines the 'id' property used to uniquely identify an object

        """

        self._gnome_id = GnomeId()

    id = property(lambda self: self._gnome_id.id)
