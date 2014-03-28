"""
schema_validation.py: Web services for validating data against a schema.

The purpose of these services is to allow the client to validate data without
attempting to save it.
"""
from cornice.resource import resource, view

from gnome.movers.random_movers import RandomMoverSchema

from webgnome import util
from webgnome.schema import (
    ModelSchema,
    WindMoverSchema,
    PointSourceReleaseSchema,
    MapSchema,
    CustomMapSchema,
    LocationFileSchema
)

from webgnome.views.services.base import BaseResource


class ResourceValidatorMetaclass(type):
    """
    Create `post` and `put` methods on this class, each of which are marked as
    Cornice views that validate against the class's `schema`.

    Classes that use this metaclass must define a class-level `schema` value
    that is a Colander schema

    Classes may optionally define a class-level `validators` field.

    If `validators` is a dict, an item using the 'put' or 'post' key will be
    passed into :func:`cornice.resource.view` as the ``validators`` argument
    for that view.

    If `validators` is not a dict, it will be passed to
    :func:`cornice.resource.view` as the ``validators`` keyword argument for
    both the 'put' and 'post' views that the metaclass creates.
    """

    def __new__(mcs, name, bases, dct):
        schema = dct.get('schema')
        validators = dct.get('validators', {})

        for method in ('post', 'put'):
            if method in dct:
                continue
            try:
                _validators = validators.get(method, None)
            except AttributeError:
                _validators = validators

            view_fn = lambda request: {'valid': True}

            if _validators:
                _view = view(schema=schema, validators=_validators)(view_fn)
            else:
                _view = view(schema=schema)(view_fn)

            dct[method] = _view

        return super(ResourceValidatorMetaclass, mcs).__new__(
            mcs, name, bases, dct)


class BaseResourceValidator(BaseResource):
    schema = None
    __metaclass__ = ResourceValidatorMetaclass


@resource(path='/model/{model_id}/validate/model', renderer='gnome_json',
          description='Validate Model JSON.')
class ModelValidator(BaseResourceValidator):
    schema = ModelSchema


@resource(path='/model/{model_id}/validate/mover/wind', renderer='gnome_json',
          description='Validate WindMover JSON.')
class WindMoverValidator(BaseResourceValidator):
    schema = WindMoverSchema
    validators = {
        'put': util.valid_wind_id
    }


@resource(path='/model/{model_id}/validate/mover/random',
          renderer='gnome_json',
          description='Validate RandomMover JSON.')
class RandomMoverValidator(BaseResourceValidator):
    schema = RandomMoverSchema


@resource(path='/model/{model_id}/validate/spill/surface_release',
          renderer='gnome_json',
          description='Validate PointSourceRelease JSON.')
class PointSourceReleaseValidator(BaseResourceValidator):
    schema = PointSourceReleaseSchema


@resource(path='/model/{model_id}/validate/map', renderer='gnome_json',
          description='Validate Map JSON.')
class MapValidator(BaseResourceValidator):
    schema = MapSchema


@resource(path='/model/{model_id}/validate/custom_map', renderer='gnome_json',
          description='Validate CustomMap JSON.')
class CustomMapValidator(BaseResourceValidator):
    schema = CustomMapSchema


@resource(path='/model/{model_id}/validate/location_file',
          renderer='gnome_json',
          description='Validate LocationFile JSON.')
class LocationFileValidator(BaseResourceValidator):
    schema = LocationFileSchema
