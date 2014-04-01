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

        :param id: Unique Id identifying the newly created mover
                   (a UUID as a string).
                   This is used when loading an object from a persisted model
        """

        self._gnome_id = GnomeId(id=kwargs.pop('id', None))

    id = property(lambda self: self._gnome_id.id)
