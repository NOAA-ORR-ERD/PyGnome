'''
Created on Apr 11, 2013

Create a unicode filename and test with object
'''
import os
import shutil
import sys

import pytest 

from gnome import environment 

datadir = os.path.join(os.path.dirname(__file__), r"SampleData")

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

def test_ucode_char_in_filename():
    # on windows
    file_ = os.path.join(datadir, u'WindDataFromGnome.WND')
    environment.Wind(filename=file_)
    ufile = create_ucode_file(file_)
    
    if sys.platform == 'win32':
        with pytest.raises(UnicodeEncodeError):
            environment.Wind(filename=ufile)
    elif sys.platform == 'darwin':
        environment.Wind(filename=ufile)
    else:
        with pytest.raises(NotImplementedError):
            environment.Wind(filename=ufile)
    
    