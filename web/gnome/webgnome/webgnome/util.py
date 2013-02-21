"""
util.py: Utility function for the webgnome package.
"""
import argparse
import errno
import colander
import datetime
import inspect
import json
import math
import os
import shutil
import sys
import time
import uuid

from functools import wraps
from itertools import chain
from pyramid.exceptions import Forbidden
from pyramid.renderers import JSON
from hazpy.unit_conversion.unit_data import ConvertDataUnits


def make_message(type_in, text):
    """
    Create a dictionary suitable to be returned in a JSON response as a
    "message" sent to the JavaScript client.

    The client looks for "message" objects included in JSON responses that the
    server sends back on successful form submits and if one is present, and has
    a ``type`` field and ``text`` field, it will display the message to the
    user.
    """
    return dict(type=type_in, text=text)


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
    elif obj is colander.null:
        return None
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

    def __get__(self, name):
        return self._fields[name]

    def get_default_field_value(self, field):
        value = field.default

        if value is colander.null and field.children:
            fields = []
            for sub_field in field.children:
                sub_field_value = self.get_default_field_value(sub_field)
                fields.append((sub_field.name, sub_field_value))
            value = dict(fields)

        return value

    def make_value_object(self, value):
        fields = []
        for key, val in value.items():
            if isinstance(val, dict):
                val = self.make_value_object(val)
            fields.append((key, val))
        return self.ObjectValue(fields)

    def get_field_value(self, field, target=None):
        """
        Return a value for a Colander field object ``field``.

        If ``target`` is provided and is an object or dictionary, then
        attempt to look up the value by the name of ``field`` on ``target``.

        Otherwise, or if the value was not found on ``target``, use the field's
        default value if provided, falling back to None.
        """
        if isinstance(target, dict):
            value = target.get(field.name, None)
        else:
            value = getattr(target, field.name, None)

        if value is None:
            value = self.get_default_field_value(field)
        else:
            value = field.serialize(value)

        if isinstance(value, dict):
            value = self.make_value_object(value)

        return value

    def create_fields(self):
        """
        Create a field on self for each field in the given Colander schema.

        If ``obj`` was given in the constructor, use any value found for a
        field by looking it up by name on ``obj``, either as a field or a key
        in a dict-like object.

        If ``obj`` was not given, look up field defaults.
        """
        for field in self.schema.children:
            self._fields[field.name] = self.get_field_value(field, self.obj)


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
            model = Model.get(model_id)
        except Model.DoesNotExist:
            model = None

    if model is None:
        request.errors.add('body', 'model', 'Model not found.')
        request.errors.status = 404
        return

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


def valid_location_file(request):
    valid_model_id(request)
    data_dir = os.path.join(request.registry.settings.location_file_dir,
                            request.matchdict['location'])
    location_file = os.path.join(data_dir, 'location.json')

    if not os.path.exists(location_file):
        request.errors.add('body', 'location_file', 'Location file not found.')
        request.errors.status = 404
        return

    request.validated['location_dir'] = data_dir
    request.validated['location_file'] = location_file


def valid_uploaded_file(request):
    """
    A Cornice validator that returns an error if the filename specified in a
    Map resource POST does not exist in the web application's filesystem.

    `filename` should be the name of a file (not its path) that exists in
    the configured `upload_dir`. Once validated, the absolute path of the file
    will be added to the `request.validated` dictionary.
    """
    valid_model_id(request)

    if request.errors:
        return

    filename_parts = request.validated['filename'].split('/')
    abs_filename = os.path.join(request.registry.settings.upload_dir,
                                *filename_parts)

    if not os.path.exists(abs_filename):
        request.errors.add('body', 'filename',
                           'A file with that filename does not exist.')
        request.errors.status = 400

    request.validated['filename'] = abs_filename


def valid_mover_id(request):
    """
    A Cornice validator that returns a 404 if a valid mover was not found using
    an ``id`` matchdict value.
    """
    valid_model_id(request)

    if request.errors:
        return

    model = request.validated['model']

    if not request.matchdict['id'] in model.movers:
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
    try:
        spill = model.spills[request.matchdict['id']]
    except KeyError:
        request.errors.add('body', 'spill', 'Spill not found.')
        request.errors.status = 404


def valid_coordinate_pair(request):
    """
    A Cornice validator that looks for a coordinate pair sent as `lat` and
    `lon` GET parameters.
    """
    lat = request.GET.get('lat', None)
    lon = request.GET.get('lon', None)

    if lat is None:
        request.errors.add('body', 'coordinates', 'Latitude is required as '
                                                  '"lat" GET parameter.')
        request.errors.status = 400

    if lon is None:
        request.errors.add('body', 'coordinates', 'Longitude is required as '
                                                  '"lon" GET parameter.')
        request.errors.status = 400

    request.validated['coordinates'] = {
        'lat': lat,
        'lon': lon
    }


def require_model(f):
    """
    Wrap a JSON view in a precondition that ensures the user has a valid model
    ID in his or her session.

    If the key is missing or no model is found for that key, create a new model.

    This decorator works on functions and methods. It returns a method decorator
    if the first argument to the function is ``self``. Otherwise, it returns a
    function decorator.
    """
    args = inspect.getargspec(f)

    if args and args.args[0] == 'self':
        @wraps(f)
        def inner_method(self, *args, **kwargs):
            model = get_model_from_session(self.request)
            settings = self.request.registry.settings
            if model is None:
                model = settings.Model.create(
                    model_images_dir=settings.model_images_dir)
            return f(self, model, *args, **kwargs)
        wrapper = inner_method
    else:
        @wraps(f)
        def inner_fn(request, *args, **kwargs):
            model = get_model_from_session(request)
            settings = request.registry.settings
            if model is None:
                model = settings.Model.create(
                    model_images_dir=settings.model_images_dir)
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
    Get the URL path for ``filename``, a model image.

    These are files in the model's base directory.
    """
    return request.static_url('webgnome:static/%s/%s/%s' % (
        request.registry.settings['model_images_url_path'],
        model.id,
        filename))


def get_runtime():
    """
    Return the current time as a string to be used as part of the file path
    for all images generated during a model run.
    """
    return time.strftime("%Y-%m-%d-%H-%M-%S")


def delete_keys_from_dict(target_dict, keys):
    """
    Recursively delete keys in `keys` from ``target_dict``.
    """
    def walk(value):
        if isinstance(value, dict):
            delete_keys_from_dict(value, keys)
        elif hasattr(value, '__iter__'):
            for val in value:
                walk(val)

    for k in keys:
        try:
            del target_dict[k]
        except KeyError:
            pass
    for value in target_dict.values():
        walk(value)

    return target_dict


def mkdir_p(path):
    """
    Make direction at ``path`` by first creating parent directories, unless they
    already exist. Similar to `mkdir -p` functionality in Unix/Linux.

    http://stackoverflow.com/questions/600268/mkdir-p-functionality-in-python
    """
    try:
        os.makedirs(path)
    except OSError as exc: # Python >2.5
        if exc.errno == errno.EEXIST and os.path.isdir(path):
            pass
        else: raise


velocity_unit_values = list(chain.from_iterable(
    [item[1] for item in ConvertDataUnits['Velocity'].values()]))


class CleanDirectoryCommand(object):
    def __init__(self, directory, description):
        self.directory = directory
        self.description = description

    def __call__(self):
        parser = argparse.ArgumentParser(description=self.description)
        parser.add_argument('--simulate', action='store_true', dest='simulate')

        args = parser.parse_args()

        if not os.path.exists(self.directory):
            print >> sys.stderr, \
                'Directory does not exist: %s' % self.directory

        files = os.listdir(self.directory)
        files.sort()

        if not args.simulate:
            shutil.rmtree(self.directory)
            os.mkdir(self.directory)

        print 'Files and directories deleted:\n%s' % '\n'.join(files)
        print 'Total (top-level): %s' % len(files)
