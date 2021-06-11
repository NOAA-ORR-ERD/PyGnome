#!/bin/bash

# this script is tun to test the setup in the docker-compose.yml file

yum install gcc gcc-c++ libXext libSM libXrender make nano -y
cd pygnome
conda install --file conda_requirements.txt -y
cd py_gnome
# python ./setup.py develop
/bin/bash

