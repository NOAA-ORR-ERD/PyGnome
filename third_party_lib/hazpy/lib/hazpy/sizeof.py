import types, warnings

INT_SIZE = 4
FLOAT_SIZE = 4

def sizeof_strings(iterable):
    """Return the concatenated length of all the strings in 'iterable'.
       Assumes all non-string/unicode objects are iterables to be recursed.
       Numbers or other instances will probably raise TypeError.
    """
    if isinstance(iterable, basestring):
        return len(iterable)
    size = 0
    return sum(sizeof_strings(x) for x in iterable)


def sizeof(obj, name="ARG", list_types=None, dict_types=None):
    """Return the approximate size of the object in bytes.

       Because Python doesn't give us tools to measure the size, we assume
       'str' is 1 byte/char, 'unicode' is 2 bytes/char, numeric types are
       4 bytes (32 bits), and None is 0 bytes.  Obviously this ignores a
       bit of overhead, but the main use of this function is to calculate the
       minimum size an object containing large strings requires. We look in
       dict and list elements, consume iterables as if they were lists, and
       look at the attributes of other objects.

       This function still has trouble with various object types.
    """
    if   obj is None:
        return 0
    if isinstance(obj, str):
        return len(obj)
    elif isinstance(obj, unicode):
        return len(obj)    # @@MO: Should be double this?
    elif isinstance(obj, (int, long)):
        return INT_SIZE
    elif isinstance(obj, float):
        return FLOAT_SIZE
    elif isinstance(obj, (list, tuple)) or (list_types is not None and \
        isinstance(obj, list_types)) or hasattr(obj, "next"):
        size = 0
        for i, elm in enumerate(obj):
            name2 = "%s[%s]" % (name, i)
            size += sizeof(elm, name2, list_types, dict_types)
        return size
    elif isinstance(obj, dict) or (dict_types is not None and \
        isinstance(obj, dict_types)):
        size = 0
        for key, value in obj.iteritems():
            name = "%s[%r]" % (obj, key)
            size += sizeof(value)
        return size
    elif hasattr(obj, "__dict__"):
        size = 0
        for key, value in obj.__dict__.iteritems():
            name = "%s.%s'" % (obj, key)
            size += sizeof(value)
        return size
    else:
        try:
            type_name = obj.__class__.__name__
            if type_name == "type":
                type_name = obj.__name__
        except AttributeError:
            type_name = obj.__name__
        msg = "assuming size 0 for %s (type %s)" % (name, type_name)
        warnings.warn(msg)
        return 0


