"""
util.py: Utility function for the webgnome package.
"""
import datetime
import inspect
import json
import math
import time
import uuid

from functools import wraps
from itertools import chain
from pyramid.exceptions import Forbidden
from pyramid.renderers import JSON
from hazpy.unit_conversion.unit_data import ConvertDataUnits


def make_message(type, text):
    """
    Create a dictionary suitable to be returned in a JSON response as a
    "message" sent to the JavaScript client.

    The client looks for "message" objects included in JSON responses that the
    server sends back on successful form submits and if one is present, and has
    a ``type`` field and ``text`` field, it will display the message to the
    user.
    """
    return dict(type=type, text=text)


def encode_json_date(obj):
    """
    Render a :class:`datetime.datetime` or :class:`datetime.date` object using
    the :meth:`datetime.isoformat` function, so it can be properly serialized to
    JSON notation.
    """
    if isinstance(obj, datetime.datetime) or isinstance(obj, datetime.date):
        return obj.isoformat()


def json_encoder(obj):
    """
    A custom JSON encoder that handles :class:`datetime.datetime` and
    :class:`datetime.date` values, with a fallback to using the :func:`str`
    representation of an object.
    """
    date_str = encode_json_date(obj)

    if date_str:
        return date_str
    else:
        return str(obj)


def json_date_adapter(obj, request):
    """
    A wrapper around :func:`json_date_encoder` so that it may be used in a
    custom JSON adapter for a Pyramid renderer.
    """
    return encode_json_date(obj)


gnome_json = JSON(adapters=(
    (datetime.datetime, json_date_adapter),
    (datetime.date, json_date_adapter),
    (uuid.UUID, lambda obj, request: str(obj))
))


def to_json(obj, encoder=json_encoder):
    return json.dumps(obj, default=encoder)


class SchemaForm(object):
    """
    A class that creates fields on itself based on a Colander schema.

    Instances are given fields of the same name as fields on the schema. Values
    for the fields are set by looking up same-named fields or dict keys in an
    ``obj`` passed into the constructor.

    If not passed both an object and a schema, the form will use any defaults
    the schema provides for field values.

    The value of each field is passed through :func:`to_json` to prepare it
    for output in an HTML document.
    """
    class ObjectValue(object):
        def __init__(self, fields):
            for field in fields:
                self.__dict__[field[0]] = field[1]

        def __repr__(self):
            return 'ObjectValue(%s)' % (
            ','.join(['%s=%s' % (k, v) for k, v in self.__dict__.items()]))

    def __init__(self, schema, obj=None):
        self.schema = schema().bind()
        self.obj = obj
        self._fields = {}
        self.create_fields()

    def __getattr__(self, name):
        if name in self._fields:
            return self._fields[name]
        else:
            raise AttributeError(name)

    def get_field_value(self, field, parents=None):
        value = None
        parents = parents or []

        if self.obj:
            target = self.obj

            for parent in parents:
                if isinstance(target, dict):
                    target = target.get(parent, None)
                else:
                    target = getattr(target, parent, None)

            if isinstance(target, dict):
                value = target.get(field.name, None)
            elif hasattr(target, field.name):
                value = getattr(target, field.name, None)
        else:
            # Use schema default. Catch defaults of 0 by checking against None.
            if field.default is not None:
                value = field.default

        value = field.serialize(value) if value is not None else ''

        if isinstance(value, dict):
            fields = []
            for key, val in value.items():
                if isinstance(val, dict):
                    val = self.get_field_value(val)
                fields.append((key, val))
            return self.ObjectValue(fields)

        return value

    def create_fields(self):
        """
        Create a field on self for each field in the given Colander schema.

        If ``obj`` was given in the constructor, use any value found for a
        field by looking it up by name on ``obj``, either as a field or a key
        in a dict-like object.

        If ``obj`` was not given, look up the form defaults in ``self.schema``.
        """
        for field in self.schema.children:
            self._fields[field.name] = self.get_field_value(field)


def get_model_from_session(request):
    """
    Return a :class:`gnome.model.Model` if the user has a session key that
    matches the ID of a running model.
    """
    settings = request.registry.settings
    model_id = request.session.get(settings.model_session_key, None)

    try:
        model = settings.Model.get(model_id)
    except settings.Model.DoesNotExist:
        model = None

    return model


def get_form_route(request, obj, route_type):
    """
    Find a route name for ``obj`` given the type of route.

    ``route_type`` is a short-hand description like "create" or "delete" used
    as a key in the ``form_routes`` dictionary.
    """
    route = None
    form_cls = get_obj_class(obj)
    routes = request.registry.settings['form_routes'].get(form_cls, None)

    if routes:
        route = routes.get(route_type, None)

    return route


MISSING_MODEL_ERROR = {
    'error': True,
    'message': make_message('error', 'That model is no longer available.')
}


def valid_model_id(request):
    """
    A Cornice validator that returns a 404 if a valid model was not found
    in the user's session.
    """
    model = None
    model_id = request.matchdict.get('model_id', None)
    Model = request.registry.settings.Model

    if model_id:
        try:
            model = Model.get(int(model_id))
        except Model.DoesNotExist:
            model = None

    if model is None:
        request.errors.add('body', 'model', 'Model not found.')
        request.errors.status = 404

    authenticated_model_id = request.session.get(
        request.registry.settings['model_session_key'], None)

    if model.id != authenticated_model_id:
        raise Forbidden()

    request.validated['model'] = model


def valid_map(request):
    """
    A Cornice validator that returns a 404 if a map was not found for the user's
    current model.
    """
    valid_model_id(request)

    if request.errors:
        return

    model = request.validated['model']

    if not model.map:
        request.errors.add('body', 'map', 'Map not found.')
        request.errors.status = 404


def valid_mover_id(request):
    """
    A Cornice validator that returns a 404 if a valid mover was not found using
    an ``id`` matchdict value.
    """
    valid_model_id(request)

    if request.errors:
        return

    model = request.validated['model']
    mover_exists = model.has_mover(int(request.matchdict['id']))

    if not mover_exists:
        request.errors.add('body', 'mover', 'Mover not found.')
        request.errors.status = 404


def valid_spill_id(request):
    """
    A Cornice validator that returns a 404 if a valid spill was not found using
    an ``id`` matchdict value.
    """
    valid_model_id(request)

    if request.errors:
        return

    model = request.validated['model']
    spill_exists = model.has_spill(int(request.matchdict['id']))

    if not spill_exists:
        request.errors.add('body', 'spill', 'Spill not found.')
        request.errors.status = 404


def require_model(f):
    """
    Wrap a JSON view in a precondition that checks if the user has a valid
    ``model_id`` in his or her session and fails if not.

    If the key is missing or no model is found for that key, create a new model.

    This decorator works on functions and methods. It returns a method decorator
    if the first argument to the function is ``self``. Otherwise, it returns a
    function decorator.

    Instead of returning an error, should we just create a model?
    """
    args = inspect.getargspec(f)

    if args and args.args[0] == 'self':
        @wraps(f)
        def inner_method(self, *args, **kwargs):
            model = get_model_from_session(self.request)
            if model is None:
                model = self.request.registry.settings.Model.create()
            return f(self, model, *args, **kwargs)
        wrapper = inner_method
    else:
        @wraps(f)
        def inner_fn(request, *args, **kwargs):
            model = get_model_from_session(request)
            if model is None:
                model = request.registry.settings.Model.create()
            return f(request, model, *args, **kwargs)
        wrapper = inner_fn
    return wrapper


def get_obj_class(obj):
    return obj if type(obj) == type else obj.__class__


class DirectionConverter(object):
    DIRECTIONS = [
        "N",
        "NNE",
        "NE",
        "ENE",
        "E",
        "ESE",
        "SE",
        "SSE",
        "S",
        "SSW",
        "SW",
        "WSW",
        "W",
        "WNW",
        "NW",
        "NNW"
    ]

    @classmethod
    def is_cardinal_direction(cls, direction):
        return direction in cls.DIRECTIONS

    @classmethod
    def get_cardinal_name(cls, degree):
        """
        Convert an integer degree into a cardinal direction name.
        """
        idx = int(math.floor((+(degree) + 360 / 32) / (360 / 16) % 16))
        if idx:
            return cls.DIRECTIONS[idx]

    @classmethod
    def get_degree(cls, cardinal_direction):
        """
       Convert a cardinal direction name into an integer degree.
       """
        idx = cls.DIRECTIONS.index(cardinal_direction.upper())
        if idx:
            return (360.0 / 16) * idx


def get_model_image_url(request, model, filename):
    """
    Get the URL path for ``filename``.

    Paths for images generated by a model include the model ID and runtime to
    separate images generated by different runs.
    """
    return request.static_url('webgnome:static/%s/%s/%s/%s' % (
        request.registry.settings['model_images_url_path'],
        model.id,
        model.runtime,
        filename))


def get_runtime():
    """
    Return the current time as a string to be used as part of the file path
    for all images generated during a model run.
    """
    return time.strftime("%Y-%m-%d-%H-%M-%S")


velocity_unit_values = list(chain.from_iterable(
    [item[1] for item in ConvertDataUnits['Velocity'].values()]))

velocity_unit_options = [(values[1][0], values[1][0]) for label, values in
                         ConvertDataUnits['Velocity'].items()]
