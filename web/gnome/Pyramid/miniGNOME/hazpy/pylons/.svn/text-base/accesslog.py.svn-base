import pprint

def log_session(log, session):
    """A simple session dumper for debugging.

    Usage:  call in your base controller's ``.__call__`` or ``.__after__``.

    ``log`` - a function taking the same arguments as ``logging.info``.
    ``session`` - pass ``pylons.session`` here.
    """
    try:      # Assume it's a StackedObjectProxy.
        session = dict((k, v) for k, v in session._current_obj().iteritems() 
            if not k.startswith("_"))
    except AttributeError:
        pass  # Else just dump whatever it is.
    log("%s", pprint.pformat(session))


def log_access(log, request, response, user=None, error_pages=False):
    """A concise access log for development and debugging.

    Not intended for production use, so does not log remote IP.

    Usage:  call in your base controller's ``.__call__`` or ``.__after__``.

    ``log`` - a function taking the same arguments as ``logging.info``.
    ``request`` - pass ``pylons.request`` here.
    ``response`` - pass ``pylons.response`` here.
    ``user`` - username of authenticated user, or '' or None if not logged in.
    ``error_pages`` - True to log error pages; false (default) to ignore them.
        An error page is anything routed to the "error" controller.
    """
    err = request.environ["pylons.routes_dict"]["controller"] == "error"
    if err and not error_pages:
        return
    status = response.status_int
    if request.query_string:
        url = "%s?%s" % (request.path_info, request.query_string)
    else:
        url = request.path_info
    if user:
        user_info = " (%s)" % user
    else:
        user_info = ""
    log("%s %s [%s]%s", status, url, request.method, user_info)
