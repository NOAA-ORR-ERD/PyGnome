"""
util.py: Utility function for the webgnome package.
"""
import datetime
from functools import wraps


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

    Raises :class:`TypeError` if the object was not of the above types and could
    not be converted to a string.
    """
    date_str = encode_json_date(obj)

    if date_str:
        return date_str
    elif hasattr(obj, '__str__'):
        return str(obj)
    else:
        raise TypeError(
            "Could not convert object of type %s into JSON" % (type(obj)))


def json_date_adapter(obj, request):
    """
    A wrapper around :func:`json_date_encoder` so that it may be used in a
    custom JSON adapter for a Pyramid renderer.
    """
    return encode_json_date(obj)


def json_require_model(f):
    """
    Wrap a JSON view in a precondition that checks if the user has a valid
    ``model_id`` in his or her session and fails if not.

    If the key is missing or no model is found for that key, return a JSON
    object with a ``message`` object describing the error.
    """
    @wraps(f)
    def inner(request, *args, **kwargs):
        settings = request.registry.settings
        model_id = request.session.get(settings.model_session_key, None)

        try:
            model = settings.Model.get(model_id)
        except settings.Model.DoesNotExist:
            model = None

        if model is None:
            return {
                'error': True,
                'message': {
                    'type': 'error',
                    'text': 'That model is no longer available.'
                }
            }
        return f(request, model, *args, **kwargs)
    return inner


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
