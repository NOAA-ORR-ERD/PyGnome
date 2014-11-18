'''
test functions in remote_data module
'''

import os
from urllib2 import HTTPError, URLError

from gnome.utilities.remote_data import get_datafile

import pytest
from ..conftest import testdata


def test_exception():
    """
    bogus file to raise an exception - do this if we have an internet
    connection before testing it
    """

    bogus = 'bogus.txt'

    try:
        with pytest.raises(HTTPError):
            get_datafile(bogus)
    except URLError:
        'no internet connection'
        return


def test_get_datafile():
    """
    downloads CLISShio.txt to make sure get_datafile works as expected

    removes the file sample_data/CLISShio.txt after downloading it to leave it
    in clean _state
    """

    file_ = testdata['CatsMover']['tide']
    if os.path.exists(file_):
        os.remove(file_)

    num_calls = 0
    while num_calls < 2:
        # do this twice to make sure it still works as expected after file has
        # been downloaded in 1st call
        try:
            r_file = get_datafile(file_)
        except:
            return

        assert os.path.exists(r_file)
        num_calls += 1
    # clean up file_ that was downloaded just for this test
    os.remove(file_)
