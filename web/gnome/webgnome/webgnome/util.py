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

