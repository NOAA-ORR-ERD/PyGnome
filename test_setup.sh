#!/bin/bash



yum install gcc gcc-c++ netcdf-devel libXext libSM libXrender nano -y
cd pygnome
conda install --file conda_requirements_py3.txt -y
cd py_gnome
# python ./setup.py develop
/bin/bash