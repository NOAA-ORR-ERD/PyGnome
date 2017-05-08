#!/bin/sh

# simple script to test the gitHub versions
# maybe this can be adapted to a CI script
#
#  This creates a new conda eenvironemnt, pulls the
#  code from gitHub, and builds and runs GNOME and the oils library and tests.
#
#  Then it builds the docs for gh-pages
#
# This should be run outside of the source dir
# but I put it in the repo so we can keep track of it.



CONDA_ENV_NAME="test_gnome_github"

## create a conda environment:

# clear out the old one if it's there:
conda remove -y -n $CONDA_ENV_NAME --all

# create a new one
conda create  -y -n $CONDA_ENV_NAME python=2

# and activate it
source activate $CONDA_ENV_NAME

# clone the repo:
git clone https://github.com/NOAA-ORR-ERD/PyGnome.git

# install the requirements
cd PyGnome
conda install -y --file conda_requirements.txt

# install the code
cd ./py_gnome
python setup.py install

# install the oil_library with pip
pip install git+https://github.com/NOAA-ORR-ERD/OilLibrary.git

# test the oil_library
pytest --pyargs oil_library

# test pygnome!
pytest --runslow

# build the docs -- why not?
cd documentation

./build_gh_pages.sh

# And done!

