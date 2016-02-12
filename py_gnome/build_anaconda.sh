#!/bin/sh

# Script to build in develop mode under Anaconda -- requires some lib re-linking!

if [[ "$1" = "" ]] ; then
    echo "usage: ./build_anaconda.sh <build_target>"
elif [[ "$1" = "develop" ]] ; then
    python setup.py $1 --no-deps
    python re_link_for_anaconda.py 
elif [[ "$1" = "install" ]] ; then
    python setup.py $1
    python re_link_for_anaconda.py --nolocalpath
else
    echo "unknown target $1"
fi


