"""
__init__.py for the gnome package

"""


__version__ = '0.1.1'
# a few imports so that the basic stuff is there

from gnomeobject import GnomeId
from . import map
from . import spill
from . import spill_container
from . import movers
from . import environment
from . import model
from . import outputters

__all__ = [GnomeId,
           map,
           spill,
           spill_container,
           movers,
           environment,
           model,
           outputters]
