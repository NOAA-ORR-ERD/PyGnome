'''
common fixture for output_dirs required by different outputters
'''

import os
import pytest
import zipfile
from pathlib import Path


@pytest.fixture(scope='function')
def tmp_output_dir(tmpdir, request):
    '''
    return a temporary directory for outputter to dump its data
    puts it in where tmpfiles go, so good for automated testing,
    not good for looking at the results.
    '''
    name = 'output_' + request.function.__name__.lstrip('test_')
    name = tmpdir.mkdir(name).strpath

    return name


@pytest.fixture(scope='module')
def output_dir(request):
    '''
    return a default output directory for outputter to dump its data.

    this puts it in the test director locally, so the user running the test can see the output.

    each test module gets its own output dir.

    '''
    # create the dir name from the module path
    path, name = os.path.split(request.module.__file__)
    name = os.path.splitext(name)[0].removeprefix('test_')
    name = os.path.join(path, "output_"+name)
    os.makedirs(name, exist_ok=True)
    return Path(name)


@pytest.fixture(scope='function')
def output_filename(output_dir, request):
    '''
    trying to create a unique file for tests so pytest_xdist doesn't
    have issues.
    '''
    os.makedirs(output_dir, exist_ok=True)
    file_name = request.function.__name__
    extension = request.module.FILE_EXTENSION
    #  This may capture multi-processing pytests
    #  and create a new filename from the process id
    # the previous code used request._pyfuncitem._genid
    #  which is no longer available in pytest
    if request._pyfuncitem.funcargs['skip_serial'] is not None:
        file_name = "{}_{}_sample.{}".format(file_name, os.getpid(), extension)
    else:
        file_name = "{}_sample{}".format(file_name, extension)

    return os.path.join(output_dir, file_name)


def count_files_in_zip(zip_filepath):
    """
    Counts the number of files in a ZIP archive.

    Args:
        zip_filepath (str): The path to the ZIP file.

    Returns:
        int: The number of files in the ZIP archive.
             Returns -1 if the file is not found or is not a valid ZIP file.
    """
    try:
        with zipfile.ZipFile(zip_filepath, 'r') as zip_file:
            return len(zip_file.namelist())
    except FileNotFoundError:
        print(f"Error: File not found: {zip_filepath}")
        return -1
    except zipfile.BadZipFile:
         print(f"Error: Not a valid ZIP file: {zip_filepath}")
         return -1
