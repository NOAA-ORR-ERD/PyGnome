'''
common fixture for output_dirs required by different outputters
'''

import os
import pytest


@pytest.fixture(scope='function')
def tmp_output_dir(tmpdir, request):
    '''
    return a temporary directory for outputter to dump its data
    puts it in where tmpfiles go, so good for autoamted testing,
    not good for looking at the results.
    '''
    name = 'output_' + request.function.func_name
    name = tmpdir.mkdir(name).strpath

    return name

@pytest.fixture(scope='module')
def output_dir(request):
    '''
    return a default output directory for outputter to dump its data.

    this puts it in the test director locally, so the user running the test can see the output.

    each test module gets its own output dir.

    '''
    #create the dir name from the module path
    path, name = os.path.split(request.module.__file__)
    name = os.path.splitext(name)[0]
    name = os.path.join(path, "output_"+name)

    # make sure it exists
    try:
        os.mkdir(name)
    except OSError:
        pass # already there

    return name
