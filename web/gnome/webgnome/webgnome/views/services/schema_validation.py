"""
schema_validation.py: Web services for validating data against a schema.

The purpose of these services is to allow the client to validate data without
attempting to save it.
"""
from cornice.resource import resource, view
from webgnome.views.services.base import BaseResource
from webgnome import schema


class ResourceValidatorMetaclass(type):
    """
    Create `post` and `put` methods on this class, each of which are marked as
    Cornice views that validate against the class's `_schema` class, a Colander
    schema.

    Classes that use this metaclass must define a class-level `_schema` value.
    """
    def __new__(mcs, name, bases, dct):
        schema = dct.get('_schema')

        @view(schema=schema)
        def validator(self):
            return {
                'valid': True
            }

        if 'post' not in dct:
            dct['post'] = validator

        if 'put' not in dct:
            dct['put'] = validator

        return super(ResourceValidatorMetaclass, mcs).__new__(mcs, name, bases,
                                                              dct)


class BaseResourceValidator(BaseResource):
    _schema = None
    __metaclass__ = ResourceValidatorMetaclass


@resource(path='validate/model', renderer='gnome_json',
          description='Validate Model JSON.')
class ModelValidator(BaseResourceValidator):
    _schema = schema.ModelSchema


@resource(path='/validate/mover/wind', renderer='gnome_json',
          description='Validate WindMover JSON.')
class WindMoverValidator(BaseResourceValidator):
    _schema = schema.WindMoverSchema


@resource(path='/validate/mover/random', renderer='gnome_json',
          description='Validate RandomMover JSON.')
class RandomMoverValidator(BaseResourceValidator):
    _schema = schema.RandomMoverSchema


@resource(path='/validate/spill/surface_release', renderer='gnome_json',
          description='Validate SurfaceReleaseSpill JSON.')
class SurfaceReleaseSpillValidator(BaseResourceValidator):
    _schema = schema.SurfaceReleaseSpillSchema


@resource(path='/validate/map', renderer='gnome_json',
          description='Validate Map JSON.')
class MapValidator(BaseResourceValidator):
    _schema = schema.MapSchema


@resource(path='/validate/custom_map', renderer='gnome_json',
          description='Validate CustomMap JSON.')
class CustomMapValidator(BaseResourceValidator):
    _schema = schema.CustomMapSchema


@resource(path='/validate/location_file', renderer='gnome_json',
          description='Validate LocationFile JSON.')
class LocationFileValidator(BaseResourceValidator):
    _schema = schema.LocationFileSchema
