'''
Created on Mar 1, 2013
'''

from colander import (
    SchemaNode,
    MappingSchema,
    Bool,
    Float,
    String,
    )

import gnome
from gnome.persist.validators import convertable_to_seconds
from gnome.persist.base_schema import Id
from gnome.persist.extend_colander import LocalDateTime


class Mover(MappingSchema):
    on = SchemaNode(Bool(), default=True, missing=True)
    active_start = SchemaNode(LocalDateTime(), default=None, missing=None,
                              validator=convertable_to_seconds)
    active_stop = SchemaNode(LocalDateTime(), default=None, missing=None,
                             validator=convertable_to_seconds)

class UpdateWindMover(Mover):
    """
    Contains properties required by UpdateWindMover and CreateWindMover
    """
    uncertain_duration = SchemaNode(Float() )
    uncertain_time_delay = SchemaNode(Float() )
    uncertain_speed_scale = SchemaNode(Float() )
    uncertain_angle_scale = SchemaNode(Float() )
    wind_id = SchemaNode(String() )
    

class CreateWindMover(Id, UpdateWindMover):
    pass

class UpdateRandomMover(Mover):
    diffusion_coef = SchemaNode( Float() )
    
class CreateRandomMover(Id,UpdateRandomMover):
    pass
    