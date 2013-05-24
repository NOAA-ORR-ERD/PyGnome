"""Logging functions for Pylons applications.
"""
import logging

from webob import Request, Response

__all__ = ["TransLogger", "no_error_pages"]

class TransLogger(object):
    """Based on ``paste.translogger`` but more flexible.

    WARNING: Not compatible with interactive traceback due to unknown bug.
    Not recommended for use at this time.

    Logs requests in a compact format useful for development debugging.

    ``app`` is the WSGI application to be wrapped.
    ``logger_name`` is the Python logger to use.  Messages will be logged at
        priority INFO.
    ``filter_func`` is a function that takes the WSGI environ and returns
        true if the request should be logged or false if not.  The default
        value ``None`` logs all request.

    No return value.
    """
    def __init__(self, app, logger_name="access", filter_func=None):
        self.app = app
        self.filter_func = filter_func
        self.log_func = logging.getLogger(logger_name).info

    def __call__(self, environ, start_response):
        request = Request(environ)
        response = request.get_response(self.app)   # Call WSGI application.
        if (not self.filter_func) or self.filter_func(environ):
            status = response.status_int
            url = request.path_info
            if request.query_string:
                url = "%s?%s" % (request.path_info, request.query_string)
            else:
                url = request.path_info
            username = environ.get("REMOTE_USER")
            if username:
                user_info = " (%s)" % username
            else:
                user_info = ""
            format = "%s %s [%s]%s"
            self.log_func(format, status, url, request.method, user_info)
        return response(environ, start_response)


#### Filter functions for use with TransLogger ####

def no_error_pages(environ):
    """TransLogger filter func to suppress error pages

    Return False if routing variable "controller" is present and contains the
    value "error".
    """
    controller = environ["wsgiorg.routing_args"][1].get("controller")
    return controller != "error"

