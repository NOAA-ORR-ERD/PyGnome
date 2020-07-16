'''
    __init__.py for the gnome.movers package
'''
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

from future import standard_library
standard_library.install_aliases()
from builtins import *
from .movers import Mover, Process, CyMover, ProcessSchema, PyMover
from .simple_mover import SimpleMover, SimpleMoverSchema
from .wind_movers import (WindMover,
                         WindMoverSchema,
                         constant_wind_mover,
                         wind_mover_from_file,
                         GridWindMover,
                         GridWindMoverSchema,
                         IceWindMover,
                         IceWindMoverSchema)

from .ship_drift_mover import ShipDriftMover, ShipDriftMoverSchema
from .random_movers import (RandomMover,
                           RandomMoverSchema,
                           IceAwareRandomMover,
                           IceAwareRandomMoverSchema,
                           RandomMover3D,
                           RandomMover3DSchema)

from .current_movers import (CatsMover,
                            CatsMoverSchema,
                            ComponentMover,
                            ComponentMoverSchema,
                            GridCurrentMover,
                            GridCurrentMoverSchema,
                            IceMover,
                            IceMoverSchema,
                            CurrentCycleMover,
                            CurrentCycleMoverSchema)

from .vertical_movers import (RiseVelocityMover,
                             RiseVelocityMoverSchema,
                             TamocRiseVelocityMover)

from .py_wind_movers import PyWindMover, PyWindMoverSchema
from .py_current_movers import PyCurrentMover, PyCurrentMoverSchema


mover_schemas = [
    WindMoverSchema,
    GridWindMoverSchema,
    IceWindMoverSchema,
    ShipDriftMoverSchema,
    SimpleMoverSchema,
    RandomMoverSchema,
    IceAwareRandomMoverSchema,
    RandomMover3DSchema,
    CatsMoverSchema,
    ComponentMoverSchema,
    GridCurrentMoverSchema,
    IceMoverSchema,
    CurrentCycleMoverSchema,
    RiseVelocityMoverSchema,
    PyWindMoverSchema,
    PyCurrentMoverSchema
]