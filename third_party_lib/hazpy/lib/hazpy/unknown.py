__all__ = ["UNKNOWN"]

class Unknown(object):
    """Universal 'unknown' object.

       Self-evident when printed as a string.
       Evaluates to false, so interchangeable with None and '' in boolean
       expressions.
       Identifiable programmatically; e.g., to replace with styled HTML.
    """
    def __str__(self):
        return "UNKNOWN"

    def __nonzero__(self):
        return False

UNKNOWN = Unknown()
