'''
Created on Apr 11, 2013

Create a Unicode filename and test with object
'''

import os
import shutil
import sys
from pathlib import Path

import pytest

from gnome.movers import CatsMover, c_GridCurrentMover, c_GridWindMover
from gnome.environment import Wind, Tide
from gnome.utilities.remote_data import get_datafile

IS_WINDOWS = sys.platform.startswith("win")

# skip all the tests for windows for now
# print(sys.platform)
# pytestmark = pytest.mark.skipif(sys.platform.startswith("win"), reason="skip all the tests for windows for now")

from gnome.cy_gnome.cy_helpers import filename_as_bytes, read_file

OUTPUT_DIR = Path(__file__).parent / "output_dir"

LATIN_CHAR = '\u00e1'  # Latin Small Letter A With Acute (U+00E1)
HEBREW_CHAR = '\u05d1'  # Hebrew character -- outside latin-1 charset

UNICODE_TEST_CHAR = LATIN_CHAR if IS_WINDOWS else HEBREW_CHAR

def test_read_file():
    """
    makes sure the the Cython read_file function works

    so we can use it in other tests
    """
    fname = str(OUTPUT_DIR / "a_filename.txt")
    contents = "Just a little ASCII text\n"
    # write a file:
    with open(fname, 'w') as fp:
        fp.write(contents)

    read_contents = read_file(fname).decode('ascii')

    print(contents)

    assert contents == read_contents


def create_ucode_file(filename):
    """
    Function simply takes a data file path and
    copies it to a new file. The new file name has
    an accented unicode (non-ascii) character:

    Latin Small Letter A With Acute (U+00E1)

    This should be enough to know that our encoding/decoding is working.

    NOTE: it might be more robust to use a character that's not in the
    latin-1 set either, but this will do for now.
    """

    (path_, fname) = os.path.split(filename)
    (name, ext) = fname.split('.')

    # new name with unicode char

    fname = name + '_' + UNICODE_TEST_CHAR + '.' + ext
    ufile = os.path.join(path_, fname)

    shutil.copyfile(filename, ufile)

    return ufile


# tests the cython utility that converts a file path to
# bytes for use in the C code
def test_filename_as_bytes_ascii():
    """
    ascii should work on all systems
    """
    fname = Path("this") / "other"
    bfname = filename_as_bytes(fname)
    assert bfname == str(fname).encode('ascii')


def test_filename_as_bytes_latin():
    """
    latin1 characters should work on all systems
     - but may not on Windows
    """

    fname = "this/that/the/other" + LATIN_CHAR
    bfname = filename_as_bytes(fname)

    print(fname)
    print(bfname)
    # not sure what assert here, as long as there is no error
    assert True


def test_filename_as_bytes_non_latin():
    """
    characters outside of latin (codepoints > 255) may not work on Windows.
    """

    # A Hebrew character
    fname = "this/that/the/other" + HEBREW_CHAR

    # make sure it's not in the latin charset
    # so we're really testing it.
    try:
        # not sure how to get the Windows mcbs, but this is close
        fname.encode('cp1252')
        assert False, "this is a latin character -- it's not supposed to be for this test"
    except UnicodeEncodeError:
        pass
    # We can't handle non-latin filenames on Windows at this point
    if IS_WINDOWS:
        with pytest.raises(ValueError):
            bfname = filename_as_bytes(fname)
    else:
        bfname = filename_as_bytes(fname)
    
    #print(fname)
    #print(bfname)
    # not sure what assert here, as long as there is no error
        assert isinstance(bfname, bytes)

#@pytest.mark.parametrize('filename', ['simple_ascii_name.txt',
#                                      'latin_char_' + LATIN_CHAR,
#                                      'non_latin_char_' + HEBREW_CHAR,
#                                      ])
def test_read_file_ascii():
    """
    check that we can read files with various names

    NOTE: this is using PATH object
    """

    filepath = OUTPUT_DIR / 'simple_ascii_name.txt'
    print(f"testing: {filepath}")
    # create a file:
    contents = "Just a little ASCII text\n"
    with open(filepath, 'w') as fp:
        fp.write(contents)

    # can we read it back in?
    read_contents = read_file(filepath).decode('ascii')

    assert contents == read_contents


datadir = os.path.join(os.path.dirname(__file__), r"sample_data")
obj_ = [(Wind, r'WindDataFromGnome.WND'),
        (Tide, os.path.join(r'tides', r'CLISShio.txt')),
        (Tide, os.path.join(r'tides', r'TideHdr.FINAL')),
        (CatsMover, os.path.join(r'long_island_sound', r'tidesWAC.CUR'))]

def test_read_file_Latin():
    """
    check that we can read files with various names

    NOTE: this is using PATH object
    """

    filepath = OUTPUT_DIR / ('latin_char_' + LATIN_CHAR)
    print(f"testing: {filepath}")
    # create a file:
    contents = "Just a little ASCII text\n"
    with open(filepath, 'w') as fp:
        fp.write(contents)

    # can we read it back in?
    read_contents = read_file(filepath).decode('ascii')

    assert contents == read_contents


def test_read_file_Hebrew():
    """
    check that we can read files with various names

    NOTE: this is using PATH object
    """

    filepath = OUTPUT_DIR / ('non_latin_char_' + HEBREW_CHAR)
    print(f"testing: {filepath}")
    # create a file:
    contents = "Just a little ASCII text\n"
    with open(filepath, 'w') as fp:
        fp.write(contents)

    # can we read it back in?
    if IS_WINDOWS:
       with pytest.raises(ValueError):
            read_contents = read_file(filepath).decode('ascii')        
    else:
       read_contents = read_file(filepath).decode('ascii')
       assert contents == read_contents  
       
datadir = os.path.join(os.path.dirname(__file__), r"sample_data")
obj_ = [(Wind, r'WindDataFromGnome.WND'),
        (Tide, os.path.join(r'tides', r'CLISShio.txt')),
        (Tide, os.path.join(r'tides', r'TideHdr.FINAL')),
        (CatsMover, os.path.join(r'long_island_sound', r'tidesWAC.CUR'))]
        

        
@pytest.mark.parametrize('test_case', obj_)
def test_ucode_char_in_filename(test_case):

    # on windows

    file_ = get_datafile(os.path.join(datadir, test_case[1]))

    # first check that it works with file without adding special character to filename

    test_case[0](filename=file_)

    ufile = create_ucode_file(file_)
    test_case[0](filename=ufile)  # This should read valid unicode filenames

    print('{0}({1}) passed the test'.format(test_case[0], test_case[1]))

    assert True

gridmovers = [(c_GridCurrentMover,
               os.path.join(r'currents', r'ny_cg.nc'),
               None),
              (c_GridCurrentMover,
               os.path.join(r'currents', r'ny_cg.nc'),
               os.path.join(r'currents', r'NYTopology.dat')),
              (c_GridWindMover,
               os.path.join(r'winds', r'WindSpeedDirSubset.nc'),
               os.path.join(r'winds', r'WindSpeedDirSubsetTop.dat')),
              ]
@pytest.mark.skipif(IS_WINDOWS, reason = "unicode filenames don't work in windows") 
@pytest.mark.parametrize('mover_test', gridmovers)
def test_ucode_char_in_grid_mover_filename(mover_test):

    file1 = get_datafile(os.path.join(datadir, mover_test[1]))
    ufile1 = create_ucode_file(file1)
    print('file1:  ', file1)
    print('ufile1:  ', ufile1)
    print(os.path.exists(ufile1))
    if mover_test[2] is None:
        file2 = None
        ufile2 = None
    else:
        file2 = get_datafile(os.path.join(datadir, mover_test[2]))
        ufile2 = create_ucode_file(file2)
    # make sure that the ascii file it there and works
    mover_test[0](file1, file2)
    # Unicode names should work for all systems
    mover_test[0](ufile1, ufile2)

    print('{0}({1},{2}) passed the test'.format(mover_test[0],
            mover_test[1], mover_test[2]))

    assert True
