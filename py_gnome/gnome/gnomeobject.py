#!/usr/bin/env python

from uuid import uuid1, uuid4

class GnomeObject(object):
    """
    The top-level base class in which all Gnome Objects are derived.

    Any global members or functionality will go in this class.
    """
    _id = None

    @property
    def id(self):
        """
        Override this method for more exotic forms of identification.

        :return: a unique ID returned by uuid.uuid1()
        """
        if not self._id:
            self._id = uuid1()
        return self._id
