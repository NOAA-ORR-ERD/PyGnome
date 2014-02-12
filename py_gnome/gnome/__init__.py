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
from . import outputter
from . import renderer
from . import netcdf_outputter

