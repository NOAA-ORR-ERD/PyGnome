'''
Created on Mar 1, 2013
'''

from colander import (
    SchemaNode,
    MappingSchema,
    Bool,
    Float,
    String,
    TupleSchema,
    drop
    )

from gnome.persist.validators import convertible_to_seconds
from gnome.persist.base_schema import Id, WorldPoint
from gnome.persist.extend_colander import LocalDateTime


class Mover(MappingSchema):

    on = SchemaNode(Bool(), default=True, missing=True)
    active_start = SchemaNode(LocalDateTime(), missing=drop,
                             validator=convertible_to_seconds)
    active_stop = SchemaNode(LocalDateTime(), missing=drop,
                            validator=convertible_to_seconds)


class WindMoversBase(Id, Mover):
    uncertain_duration = SchemaNode(Float(), default=24)
    uncertain_time_delay = SchemaNode(Float(), default=0)
    uncertain_speed_scale = SchemaNode(Float(), default=2)
    uncertain_angle_scale = SchemaNode(Float(), default=0.4)
    uncertain_angle_units = SchemaNode(String(), default='rad')


class WindMover(WindMoversBase):

    """
    Contains properties required by UpdateWindMover and CreateWindMover
    """
    # only used to create new WindMover
    wind_id = SchemaNode(String(), missing=drop)


class RandomMover(Id, Mover):
    diffusion_coef = SchemaNode(Float())


class SimpleMoverVelocity(TupleSchema):
    vel_x = SchemaNode(Float())
    vel_y = SchemaNode(Float())
    vel_z = SchemaNode(Float())


class SimpleMover(Id, Mover):
    uncertainty_scale = SchemaNode(Float())
    velocity = SimpleMoverVelocity()


class CatsMover(Id, Mover):
    """
    Contains properties required by UpdateWindMover and CreateWindMover
    """
    filename = SchemaNode(String(), missing=drop)
    scale = SchemaNode(Bool())
    scale_refpoint = WorldPoint()
    scale_value = SchemaNode(Float())
    # can have CatsMover without Tide object
    tide_id = SchemaNode(String(), missing=drop)


class GridCurrentMover(Id, Mover):
    filename = SchemaNode(String(), missing=drop)
    topology_file = SchemaNode(String(), missing=drop)


class GridWindMover(WindMoversBase):

    """ Similar to WindMover except it doesn't have wind_id"""

    wind_file = SchemaNode(String(), missing=drop)
    topology_file = SchemaNode(String(), missing=drop)
