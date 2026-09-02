'''
    __init__.py for the gnome.movers package
'''
# ruff: noqa: I001  -- movers.py must load first; all other submodules import CyMover/Mover/ProcessSchema from gnome.movers
from .movers import CyMover, Mover, Process, ProcessSchema, PyMover
from .simple_mover import SimpleMover, SimpleMoverSchema
from .c_wind_movers import (
    IceWindMover,
    IceWindMoverSchema,
    PointWindMover,
    PointWindMoverSchema,
    c_GridWindMover,
    c_GridWindMoverSchema,
    constant_point_wind_mover,
    point_wind_mover_from_file,
)
from .ship_drift_mover import ShipDriftMover, ShipDriftMoverSchema
from .random_movers import (
    IceAwareRandomMover,
    IceAwareRandomMoverSchema,
    RandomMover,
    RandomMover3D,
    RandomMover3DSchema,
    RandomMoverSchema,
)
from .c_current_movers import (
    CatsMover,
    CatsMoverSchema,
    ComponentMover,
    ComponentMoverSchema,
    CurrentCycleMover,
    CurrentCycleMoverSchema,
    IceMover,
    IceMoverSchema,
    c_GridCurrentMover,
    c_GridCurrentMoverSchema,
)
from .vertical_movers import (
    RiseVelocityMover,
    RiseVelocityMoverSchema,
    TamocRiseVelocityMover,
)
from .py_wind_movers import WindMover, WindMoverSchema
from .py_current_movers import CurrentMover, CurrentMoverSchema

mover_schemas = [
    PointWindMoverSchema,
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
    WindMoverSchema,
    CurrentMoverSchema
]
