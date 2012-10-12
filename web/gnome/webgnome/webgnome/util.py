"""
util.py: Utility function for the webgnome package.
"""
from functools import wraps


def json_date_adapter(obj, request):
    """
    Render a `datetime.datetime` or `datetime.date` object using the
    `isoformat()` function, so it can be properly serialized to JSON notation.

    TODO: Move to a `utils` module.
    """
    return obj.isoformat()


def json_require_model(f):
    """
    Wrap a JSON view in a precondition that checks if the user has a valid
    `model_id` in his or her session and fails if not.

    If the key is missing or no model is found for that key, return a JSON
    object with a `message` object describing the error.
    """
    @wraps(f)
    def inner(request, *args, **kwargs):
        settings = request.registry.settings
        model_id = request.session.get(settings.model_session_key, None)
        model = settings.running_models.get(model_id)

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


