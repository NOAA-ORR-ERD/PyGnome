"""
experimental code for a class decorator that automatically
sets the schema for an object for serialization

So far -- does not handle subclassing, which is a BIG missing feature

NOTE: maybe it would be better to do with a metaclass
      for GNOME_ID ?
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals


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
    # find all class attributes that are colander "nodes"
    nodes = {name: node for name, node in list(cls.__dict__.items()) if
             isinstance(node, colander.SchemaType)
             }

    # remove the nodes from the class
    for name in nodes:
        delattr(cls, name)
    name = cls.__name__ + str("Schema")
    schema = type(name, (base_schema.ObjTypeSchema,), nodes) # str hack to support py2

    cls._schema = schema

    return cls



