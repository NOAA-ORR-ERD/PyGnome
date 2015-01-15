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
import os
from subprocess import check_call, check_output

# find the path to libcurl
conda_info = check_output('conda info', shell=True)

conda_path = conda_info.split("default environment :")[1].split()[0].strip()

lib_name = 'libcurl.4.dylib'

lib_path = os.path.join(conda_path, 'lib', lib_name)

# install_name_tool -change libcurl.4.dylib /Users/chris.barker/anaconda/envs/py_gnome/lib/libcurl.4.dylib cy_basic_types.so
command = "install_name_tool -change %s %s gnome/cy_gnome/cy_basic_types.so"%(lib_name, lib_path)
check_call(command, shell=True)


