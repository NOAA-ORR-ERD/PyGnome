'''
Created on Mar 1, 2013
'''

from colander import (SchemaNode, drop,
                      MappingSchema, TupleSchema,
                      Bool, Float, String)

from gnome.persist.validators import convertible_to_seconds
from gnome.persist.base_schema import Id
from gnome.persist.extend_colander import LocalDateTime


class Weatherer(Id):
    on = SchemaNode(Bool(), default=True, missing=True)
    active_start = SchemaNode(LocalDateTime(), missing=drop,
                              validator=convertible_to_seconds)
    active_stop = SchemaNode(LocalDateTime(), missing=drop,
                             validator=convertible_to_seconds)
