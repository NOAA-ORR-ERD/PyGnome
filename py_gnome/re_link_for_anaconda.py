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

from subprocess import check_call, check_output


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
    lib_path = os.path.join(sys.exec_prefix, 'lib', lib_name)
    if not os.path.isfile(lib_path):
        # if we are in an anaconda virtual environment, we might need to
        # strip the last two folders (envs/<env_name>) to get to the main
        # anaconda lib file.
        base_exec_prefix = os.sep.join(sys.exec_prefix.split(os.sep)[:-2])
        lib_path = os.path.join(base_exec_prefix, 'lib', lib_name)

    return lib_path


def find_cy_gnome_library(shared_lib):
    return os.path.join(imp.find_module('gnome')[1], 'cy_gnome', shared_lib)


def main(argv):
    try:
        optlist, _args = getopt.getopt(argv[1:], '', ['nolocalpath'])
    except getopt.GetoptError:
        sys.exit(2)

    for opt, _arg in optlist:
        if opt in ("--nolocalpath",):
            remove_local_dir_from_path()

    # find the path to our .so dependencies
    for lib_name in ('libcurl.4.dylib', 'libstdc++.6.dylib'):
        lib_path = get_conda_lib_path(lib_name)
        cy_gnome_library = find_cy_gnome_library('cy_basic_types.so')

        command = ('otool -L {}' .format(cy_gnome_library))

        old_lib_name = None
        for l in check_output(command, shell=True).split('\n'):
            matching_tokens = [t for t in l.split() if lib_name in t]
            if len(matching_tokens) > 0:
                old_lib_name = matching_tokens[0]
                break

        if old_lib_name is None:
            # TODO: if we don't have an old rpath, then our re-link command
            # will probably fail.  But this is not very likely.
            # we will worry about that later
            old_lib_name = lib_name

        command = ('install_name_tool -change {} {} {}'
                   .format(old_lib_name, lib_path, cy_gnome_library))

        check_call(command, shell=True)


if __name__ == "__main__":
    if sys.version_info.major >= 3:
        print("Relinking is not required for Py3 -- try using plain setup.py")
    else:
        main(sys.argv)
