'''
common fixture for output_dirs required by different outputters
'''
import os

import pytest


@pytest.fixture(scope='module')
def output_dir(dump, request):
    '''
    return a default output directory for outputter to dump its data
    construct an output directory name from request object
    write it as module scope since the outputter should rewind and clean out
    the directory before running model
    '''
    d_name = request.fspath.purebasename.split('_')[1] + '_outputdir'
    odir = os.path.join(dump, d_name)
    if not os.path.isdir(odir):
        os.mkdir(odir)
    return odir
