'''
    __init__.py for the gnome.movers package
'''





from .movers import Mover, Process, CyMover, ProcessSchema, PyMover
from .simple_mover import SimpleMover, SimpleMoverSchema
from .c_wind_movers import (WindMover,
                         WindMoverSchema,
                         constant_point_wind_mover,
                         point_wind_mover_from_file,
                         c_GridWindMover,
                         c_GridWindMoverSchema,
                         IceWindMover,
                         IceWindMoverSchema)

from .ship_drift_mover import ShipDriftMover, ShipDriftMoverSchema
from .random_movers import (RandomMover,
                           RandomMoverSchema,
                           IceAwareRandomMover,
                           IceAwareRandomMoverSchema,
                           RandomMover3D,
                           RandomMover3DSchema)

from .c_current_movers import (CatsMover,
                            CatsMoverSchema,
                            ComponentMover,
                            ComponentMoverSchema,
                            c_GridCurrentMover,
                            c_GridCurrentMoverSchema,
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
    c_GridWindMoverSchema,
    IceWindMoverSchema,
    ShipDriftMoverSchema,
    SimpleMoverSchema,
    RandomMoverSchema,
    IceAwareRandomMoverSchema,
    RandomMover3DSchema,
    CatsMoverSchema,
    ComponentMoverSchema,
    c_GridCurrentMoverSchema,
    IceMoverSchema,
    CurrentCycleMoverSchema,
    RiseVelocityMoverSchema,
    PyWindMoverSchema,
    PyCurrentMoverSchema
]