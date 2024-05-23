"""
experimental code for a class decorator that automatically
sets the schema for an object for serialization

NOTE: maybe it would be better to do with a metaclass
      for GNOME_ID ?
"""

# from future import standard_library
# standard_library.install_aliases()
# from builtins import *


def serializable(cls):
    """
    make a class serializable

    this decorator finds all class attributes
    that are colander.SchemaType -- and adds them to the schema
    """
    print(cls.__name__)
    type("Schema")
    Schema(base_schema.ObjTypeSchema)
    for _attr in cls.__dict__:
        pass
