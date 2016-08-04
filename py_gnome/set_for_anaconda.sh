#!/bin/bash

# kludge to get library path right with Anaconda
# this should be written (correctly) to find teh right path
# if installed with conda build, then this wouldn't be required

# this file needs to be "sourced" at the command line:
# $ source set_for_anaconda.sh

export DYLD_LIBRARY_PATH=/Users/chris.barker/PythonStuff/Anaconda/anaconda/lib


