'''
common fixture for output_dirs required by different outputters
'''

import os
import pytest

# kludge to get a unique ID for each file
# just in case they are running in parallel
#  This may not work if they are a separate process ...

def get_id():
    id = 0
    while True:
        id += 1
        yield id

id = get_id()


@pytest.fixture(scope='function')
def tmp_output_dir(tmpdir, request):
    '''
    return a temporary directory for outputter to dump its data
    puts it in where tmpfiles go, so good for automated testing,
    not good for looking at the results.
    '''
    name = 'output_' + request.function.func_name.lstrip('test_')
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
    name = os.path.splitext(name)[0].lstrip('test_')
    name = os.path.join(path, "output_"+name)

    # make sure it exists
    try:
        os.mkdir(name)
    except OSError:
        pass # already there

    return name

@pytest.fixture(scope='function')
def output_filename(output_dir, request):
    '''
    trying to create a unique file for tests so pytest_xdist doesn't have
    issues.
    '''
    dirname = output_dir
    if not os.path.exists(dirname):
        os.mkdir(dirname)

    file_name = request.function.func_name
    # # ._genid is an internal attribute, and no longer there in pytest
    # if request._pyfuncitem._genid is None:
    #     file_name += '_sample.nc'
    # else:
    #     file_name += '_' + request._pyfuncitem._genid + '_sample.nc'

    file_name = "{}_{}_sample.nc".format(file_name, next(id))

    return os.path.join(dirname, file_name)
