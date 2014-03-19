'''
Created on Mar 1, 2013
'''

from colander import (SchemaNode, TupleSchema, MappingSchema,
                      Bool, String, Float,
                      drop)

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
    uncertain_angle_units = SchemaNode(String(), default='rad', missing=drop)


class WindMover(WindMoversBase):
    """
    Contains properties required by UpdateWindMover and CreateWindMover
    """
    # 'wind' schema node added dynamically
    name = 'WindMover'
    description = 'wind mover properties'


class RandomMover(Id, Mover):
    diffusion_coef = SchemaNode(Float())


class RandomVerticalMover(Id, Mover):
    vertical_diffusion_coef_above_ml = SchemaNode(Float())
    vertical_diffusion_coef_below_ml = SchemaNode(Float())
    mixed_layer_depth = SchemaNode(Float())


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
    scale_refpoint = WorldPoint(missing=drop)
    scale_value = SchemaNode(Float())


class GridCurrentMover(Id, Mover):
    filename = SchemaNode(String(), missing=drop)
    topology_file = SchemaNode(String(), missing=drop)


class GridWindMover(WindMoversBase):
    """ Similar to WindMover except it doesn't have wind_id"""
    wind_file = SchemaNode(String(), missing=drop)
    topology_file = SchemaNode(String(), missing=drop)


class RiseVelocityMover(Id, Mover):
    water_density = SchemaNode(Float())
    water_viscosity = SchemaNode(Float())
