#!/usr/bin/env python

"""
Script to re-write the gnome extansions to link against the correct libcurl
(i.e. the Anaconda one, not the system one, as that gets linked at bulid time.)

This script need to be run each time you re-build the extsnsions if you are
working in an Anaconda environment, and using setup.py develop mode.

(It should be updated to support regular setup.py install, too.)


On OS-X, this runs install_name_tool to specify the conda environment lib.

On Windows -- maybe something needs to be done -- not sure!

"""
import sys
import os
import getopt
import imp

from subprocess import check_call


def remove_local_dir_from_path():
    '''
        OK, why would we want to do this?

        Well if we performed a 'setup.py install', then a ,egg-info should have
        been delivered to the anaconda site-packages.
        In this case, when trying to re-link our shared objects, we would
        really like to query the installed gnome module, and not the local
        gnome sources.  And we can't do that if the local path exists
        in sys.path.  So we will remove it from the list.
    '''
    script_dir = os.path.dirname(os.path.realpath(__file__))

    for i, p in reversed(list(enumerate(sys.path))):
        if p == script_dir:
            del sys.path[i]


def get_conda_lib_path(lib_name):
    return os.path.join(sys.exec_prefix, 'lib', lib_name)


def find_cy_gnome_library(shared_lib):
    return os.path.join(imp.find_module('gnome')[1], 'cy_gnome', shared_lib)


def main(argv):
    try:
        optlist, args = getopt.getopt(argv[1:], '', ['nolocalpath'])
    except getopt.GetoptError:
        sys.exit(2)

    for opt, arg in optlist:
        if opt in ("--nolocalpath",):
            remove_local_dir_from_path()

    # find the path to libcurl
    # Note: Should we implement a list of libraries that we can iterate over?
    lib_name = 'libcurl.4.dylib'
    lib_path = get_conda_lib_path(lib_name)
    cy_gnome_library = find_cy_gnome_library('cy_basic_types.so')

    command = ('install_name_tool -change {} {} {}'
               .format(lib_name, lib_path, cy_gnome_library))

    check_call(command, shell=True)


if __name__ == "__main__":
    main(sys.argv)
