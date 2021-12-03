'''
Created on Apr 11, 2013

Create a unicode filename and test with object
'''






import os
import shutil
import sys

import pytest

from gnome.movers import CatsMover, c_GridCurrentMover, GridWindMover
from gnome.environment import Wind, Tide
from gnome.utilities.remote_data import get_datafile


def create_ucode_file(filename, valid=True):
    """
    Function simply takes a data file with path
    copies it to a new file. The new file has an accented unicode
    character.

    The boolean flag valid either uses a valid unicode (\\u00e1) character or
    an invalid one (\\u0301).
    They are both valid on the Mac since it uses UTF-8; however, on windows,
    the defalut encoding is cp1252, so it should raise the
    UnicodeEncodeError for an invalid flag
    """

    (path_, fname) = os.path.split(filename)
    (name, ext) = fname.split('.')

    # new name with unicode char

    if valid:
        fname = name + '_' + '\u00e1' + '.' + ext
    else:
        fname = name + '_' + 'a\u0301' + '.' + ext

    ufile = os.path.join(path_, fname)
    shutil.copyfile(filename, ufile)
    return ufile


datadir = os.path.join(os.path.dirname(__file__), r"sample_data")
obj_ = [(Wind, r'WindDataFromGnome.WND'), (Tide, os.path.join(r'tides',
        r'CLISShio.txt')), (Tide, os.path.join(r'tides',
        r'TideHdr.FINAL')), (CatsMover,
        os.path.join(r'long_island_sound', r'tidesWAC.CUR'))]


@pytest.mark.parametrize('test_case', obj_)
def test_ucode_char_in_filename(test_case):

    # on windows

    file_ = get_datafile(os.path.join(datadir, test_case[1]))

    # first check that it works with file without adding special character to filename

    test_case[0](filename=file_)

    ufile = create_ucode_file(file_)
    test_case[0](filename=ufile)  # This should read valid unicode filenames

    invalid_ufile = create_ucode_file(file_, False)
    if sys.platform == 'win32':
        with pytest.raises(UnicodeEncodeError):
            test_case[0](filename=invalid_ufile)
    elif sys.platform == 'darwin':
        test_case[0](filename=invalid_ufile)

    print('{0}({1}) passed the test'.format(test_case[0], test_case[1]))
    assert True

gridmover_ = [(c_GridCurrentMover, os.path.join(r'currents', r'ny_cg.nc'
              ), None), (GridWindMover, os.path.join(r'winds',
              r'WindSpeedDirSubset.nc'), os.path.join(r'winds',
              r'WindSpeedDirSubsetTop.dat')), (c_GridCurrentMover,
              os.path.join(r'currents', r'ny_cg.nc'),
              os.path.join(r'currents', r'NYTopology.dat'))]


@pytest.mark.parametrize('mover_test', gridmover_)
def test_ucode_char_in_grid_mover_filename(mover_test):

    # on windows

    file1_ = get_datafile(os.path.join(datadir, mover_test[1]))
    ufile1 = create_ucode_file(file1_)

    # invalid unicode for windows
    invalid_ufile1 = create_ucode_file(file1_, valid=False)

    if mover_test[2] is None:
        file2_ = None
        ufile2 = None
        invalid_ufile2 = None
    else:
        file2_ = get_datafile(os.path.join(datadir, mover_test[2]))
        ufile2 = create_ucode_file(file2_)

        # invalid unicode for windows
        invalid_ufile2 = create_ucode_file(file2_, valid=False)

    mover_test[0](file1_, file2_)

    # valid unicode names should work for both systems
    mover_test[0](ufile1, ufile2)

    if sys.platform == 'win32':
        with pytest.raises(UnicodeEncodeError):
            mover_test[0](invalid_ufile1, invalid_ufile2)
    elif sys.platform == 'darwin':
        # todo: also need a test case for linux2
        mover_test[0](invalid_ufile1, invalid_ufile2)

    print('{0}({1},{2}) passed the test'.format(mover_test[0],
            mover_test[1], mover_test[2]))
    assert True
