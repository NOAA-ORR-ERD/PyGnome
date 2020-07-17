"""
enum.py

An extension to the enum module that provides some extra features
"""

from __future__ import print_function
from __future__ import division
from __future__ import absolute_import
from __future__ import unicode_literals


import enum

# just so it's here
Enum = enum.Enum

class IntEnum(enum.IntEnum):

    @classmethod
    def names(cls):
        return tuple(cls.__members__)

    @classmethod
    def values(cls):
        return tuple(cls.__members__.values())


