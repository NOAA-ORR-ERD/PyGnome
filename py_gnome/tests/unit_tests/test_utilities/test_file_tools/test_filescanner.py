#!/usr/bin/env python

"""
Tests for the filescanner module

Designed to be run with py.test
"""
import pytest
import numpy as np

from gnome.utilities.file_tools.filescanner import scan

# write a test file with various separators.
tiny_file = "junk_tiny.txt"
open(tiny_file, 'w').write("""23.4, 45.6, 354
13.23,  45.6
2.3   5.4   3.4
some random text
14;  32  ;  3

14.2

now more at the end of the file
""")
# The expected results when scanned
tiny_arr = np.array((23.4, 45.6, 354, 13.23, 45.6, 2.3, 5.4, 3.4, 14, 32, 3, 14.2), dtype=np.float64)

def test_call():
    " can we even call it"
    f = open(tiny_file)
    result = scan(f, 10)
    print result
    assert True

def test_scan_n():
    f = open(tiny_file)
    result = scan(f, 10)
    assert np.array_equal(tiny_arr[:10], result)

def test_scan_not_enough():
    f = open(tiny_file)
    with pytest.raises(ValueError):
        result = scan(f, 15)


def test_scan_leave_file_in_right_place():
    f = open(tiny_file)
    result = scan(f, 10)
    next = f.readline()
    assert next.strip() == ";  3"

def test_scan_leave_file_in_right_place2():
    f = open(tiny_file)
    result = scan(f, 8)
    assert np.array_equal(result, tiny_arr[:8])
    next = f.readline()
    print next
    print "rest of file:\n", f.read()
    assert next.strip() == "some random text"


def test_scan_all():
    f = open(tiny_file)
    result = scan(f)
    print result.shape
    assert np.array_equal(result, tiny_arr)

def test_assert_not_file():
    with pytest.raises(TypeError):
        f = scan('a_string', 4)

def test_file_closed():
    f = open(tiny_file)
    f.close()
    with pytest.raises(TypeError):
        result = scan(f, 10)

def test_wrong_file_mode():
    f = open('junk.txt', 'w')
    with pytest.raises(TypeError):
        result = scan(f, 10)

def test_read_zero_vals():
    # it might as well work
    f = open(tiny_file)
    result = scan(f, 0)
    assert np.array_equal(result, np.zeros((0,)))

def test_read_negative_vals():
    # it might as well work
    f = open(tiny_file)
    with pytest.raises(OverflowError):
        result = scan(f, -3)


def test_assert_not_int():
    f = open(tiny_file)
    with pytest.raises(TypeError):
        result = scan(f, 'a_string')

def test_big():
    """
    test scanning a large enough data file to need re-allocating arrays
    """
    N = 10000
    f = open('junk_large.txt', 'w')
    for i in range(N):
        f.write("%f,\n"%3.1459)
    f.close()
    f = open('junk_large.txt', 'r')
    result = scan(f, num_to_read=None)

    assert np.array_equal( result, np.zeros((N,), dtype=np.float64)+3.1459 )


if __name__ == "__main__":
    test_call()
    # test_assert_not_int()



