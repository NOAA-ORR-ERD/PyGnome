#!/usr/bin/env python

class GnomeObject(object):
    """
    The top-level base class in which all Gnome Objects are derived.

    Any global members or functionality will go in this class.
    """
    @property
    def id(self):
        """
        Override this method for more exotic forms of identification.

        :return: the integer ID returned by the builtin id() for this object
        """
        return id(self)
