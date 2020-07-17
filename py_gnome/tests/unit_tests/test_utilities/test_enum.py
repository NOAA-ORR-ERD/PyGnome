"""
tests for the gnome enums
"""

from __future__ import print_function
from __future__ import division
from __future__ import absolute_import
from __future__ import unicode_literals


from gnome.utilities.enum import IntEnum


def test_ordinary():
    class Test(IntEnum):
        this = 1
        that = 2

    assert Test.this == 1
    assert Test.that == 2


def test_names():
    class Test(IntEnum):
        this = 1
        that = 2

    assert Test.names() == ("this", "that")


def test_values():
    class Test(IntEnum):
        this = 1
        that = 2

    assert Test.values() == (1, 2)

