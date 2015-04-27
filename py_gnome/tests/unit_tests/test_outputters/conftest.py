'''
common fixture for output_dirs required by different outputters
'''

import pytest


@pytest.fixture(scope='function')
def output_dir(tmpdir, request):
    '''
    return a default output directory for outputter to dump its data
    construct an output directory name from request object
    '''
    name = 'output_' + request.function.func_name
    name = tmpdir.mkdir(name).strpath

    return name
