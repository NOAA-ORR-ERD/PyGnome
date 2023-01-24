'''
Download data from remote server
'''

import os

from urllib.parse import urljoin
from urllib.request import urlopen
from urllib.error import HTTPError

from progressbar import (ProgressBar, Percentage, FileTransferSpeed,
                         ETA, Bar)

data_server = 'https://gnome.orr.noaa.gov/py_gnome_testdata/'

CHUNKSIZE = 1024 * 1024


def get_datafile(filename):
    """
    Looks to see if filename exists in local directory. If it exists,
    then it simply returns the 'filename' back as a string.

    If 'filename' does not exist in the local filesystem, then it tries to
    download it from the gnome server
    (http://gnome.orr.noaa.gov/py_gnome_testdata).
    If it successfully downloads the file, it puts it in the user specified
    path given in filename and returns the 'filename' string.

    If file is not found or server is down, it re-throws the HTTPError raised
    by urllib

    :param filename: path to the file including filename
    :type filename: string

    :exception: raises HTTPError if server is down or file not found
                on server

    :returns: returns the string 'filename' once it has been downloaded to
              user specified location

    """

    if os.path.exists(filename):
        return filename
    else:

        # download file, then return filename path

        (path_, fname) = os.path.split(filename)
        if path_ == '':
            path_ = '.'     # relative to current path

        try:
            resp = urlopen(urljoin(data_server, fname))
        except HTTPError as ex:
            ex.msg = ("{0}. '{1}' not found on server or server is down"
                      .format(ex.msg, fname))
            raise ex

        # progress bar
        widgets = [fname + ':      ',
                   Percentage(),
                   ' ',
                   Bar(),
                   ' ',
                   ETA(),
                   ' ',
                   FileTransferSpeed(),
                   ]

        pbar = ProgressBar(widgets=widgets,
                           maxval=int(resp.info()['Content-Length'])
                           ).start()

        if not os.path.exists(path_):
            os.makedirs(path_)

        sz_read = 0
        with open(filename, 'wb') as fh:
            # while sz_read < resp.info().getheader('Content-Length')
            # goes into infinite recursion so break loop for len(data) == 0
            while True:
                data = resp.read(CHUNKSIZE)

                if len(data) == 0:
                    break
                else:
                    fh.write(data)
                    sz_read += len(data)

                    if sz_read >= CHUNKSIZE:
                        pbar.update(CHUNKSIZE)

        pbar.finish()
        return filename
