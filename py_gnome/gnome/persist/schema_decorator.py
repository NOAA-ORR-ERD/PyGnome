"""
experimental code for a class decorator that automatically
sets the schema for an object for serialization

NOTE: maybe it would be better to do with a metaclass
      for GNOME_ID ?
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

from future import standard_library
standard_library.install_aliases()
from builtins import *
import colander
from gnome.persist import base_schema

def serializable(cls):
    """
    make a class serializable

    this decorator finds all class attributes
    that are colander.SchemaType -- and adds them to the schema
    """
    print(cls.__name__)
    nodes = {}
    print(cls.__dict__)
    print(type(cls.__dict__))
    nodes = {name: node for name, node in list(cls.__dict__.items()) if
             isinstance(node, colander.SchemaType)
             }
    cls.__dict__ = {name: val for name, val in list(cls.__dict__.items()) if
                    not isinstance(val, colander.SchemaType)
                    }
    name = cls.__name__ + "Schema"
    schema = type(name, (base_schema.ObjTypeSchema,), nodes)

    cls._schema = schema

    return cls



