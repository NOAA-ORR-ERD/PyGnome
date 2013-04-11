'''
Created on Apr 11, 2013

Create a unicode filename and test with object
'''
import os
import shutil
import sys

import pytest 

from gnome import environment, movers

def create_ucode_file(filename):
    """
    Function simply takes 
    """
    path_, fname = os.path.split(filename)
    name, ext = fname.split('.')
    fname = name+'_'+u'a\u0301'+'.'+ext # new name with unicode char
    ufile = os.path.join(path_, fname)
    shutil.copyfile( filename, ufile )
    return ufile


# join datadir to path in obj_ inside the test
datadir = os.path.join(os.path.dirname(__file__), r"SampleData")
obj_ = [(environment.Wind, r'WindDataFromGnome.WND'),
        (environment.Tide, os.path.join('tides',r'CLISShio.txt')),
        (environment.Tide, os.path.join('tides',r'TideHdr.FINAL'))]

@pytest.mark.parametrize("test_case", obj_)
def test_ucode_char_in_filename(test_case):
    # on windows
    file_ = os.path.join(datadir, test_case[1])
    ufile = create_ucode_file(file_)
    
    # first check that it works with file without adding special character to filename
    test_case[0](filename=file_)
    
    if sys.platform == 'win32':
        with pytest.raises(UnicodeEncodeError):
            test_case[0](filename=ufile)
    elif sys.platform == 'darwin':
        test_case[0](filename=ufile)
    else:
        with pytest.raises(NotImplementedError):
            test_case[0](filename=ufile)
    
    print "{0}({1}) passed the test".format(test_case[0], test_case[1])
    assert True


gridmover_ = [(movers.GridCurrentMover, os.path.join(r'currents',r'ny_cg.nc'), None),
              (movers.GridCurrentMover, os.path.join(r'winds',r'test_wind.cdf'), os.path.join(r'winds',r'WindSpeedDirSubsetTop.DAT')),
              (movers.GridWindMover, os.path.join(r'currents',r'ny_cg.nc'), os.path.join(r'currents',r'NYTopology.dat')),
              ]

@pytest.mark.parametrize("mover_test", gridmover_)
def test_ucode_char_in_grid_mover_filename(mover_test):
    # on windows
    file1_ = os.path.join(datadir, mover_test[1])
    ufile1 = create_ucode_file(file1_)
    
    if mover_test[2] is None:
        file2_ = None
        ufile2 = None
    else:
        file2_ = os.path.join(datadir, mover_test[2])
        ufile2 = create_ucode_file(file2_)
    
    # first check that it works with file without adding special character to filename
    print mover_test
    mover_test[0](file1_, file2_)
    
    if sys.platform == 'win32':
        with pytest.raises(UnicodeEncodeError):
            mover_test[0](ufile1, ufile2)
    elif sys.platform == 'darwin':
        mover_test[0](ufile1, ufile2)
    else:
        with pytest.raises(NotImplementedError):
            mover_test[0](ufile1, ufile2)
    
    print "{0}({1},{2}) passed the test".format(mover_test[0], mover_test[1], mover_test[2])
    assert True
    
