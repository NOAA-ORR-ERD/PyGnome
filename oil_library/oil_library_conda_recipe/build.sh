#!/bin/bash

# from the Anaconda mailing list: set the conda build dir to the local source

SRC_DIR=$RECIPE_DIR/..

cd $SRC_DIR

echo "current directory for building"
echo `pwd`
#$PYTHON setup.py --quiet install 
$PYTHON setup.py install

echo "after the python command"



