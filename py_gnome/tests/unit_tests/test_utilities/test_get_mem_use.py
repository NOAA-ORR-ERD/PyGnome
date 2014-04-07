#!/usr/bin/env python

"""
unit tests for the memory use check

not sure how to test for real, but at least this tells you that you can call them...

designed to be run with py.test
"""

from gnome.utilities import get_mem_use

def test_mem_use():
    """
    can we call it?
    """
    val_KB = get_mem_use('KB')
    val_MB = get_mem_use()
    val_GB = get_mem_use('GB')

    assert val_MB == val_KB / 1024

    assert val_GB == val_KB / 1024 / 1024

def test_increase():
    """
    does it go up when you allocate objects?

    Note: this may not pass if the python process has a bunch of spare memory allocated already..
    """
    import array
    start = get_mem_use()
    l = [array.array('b', b'some bytes'*1024) for i in xrange(10000)]

    assert get_mem_use() > start