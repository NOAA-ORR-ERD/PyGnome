#!/usr/bin/env python

"""
conda_pip.py

Kludgy script to grab a package from pip, make a conda package out of it,
build the conda packge, push it up to binstar, then install it.

This is helpful for making sure that all your packages are managed by conda

NOTE: On Windows, you need the conda "patch" package for this to work:
      $ conda install patch

NOTE: This will add both your binstar channel, and Aaron Meurer's channel
    -- he's a core conda dev and has a lot of semi-experimental stuff.

Set you channel by setting BINSTAR_USE below:
""" 

# hard coded binstar channel -- update for your needs
BINSTAR_USER="NOAA-ORR-ERD"

import sys, os
from subprocess import call, check_call, check_output, CalledProcessError, STDOUT 

def add_binstar_channel(USER):
    """
    adds the binstar channel to conda configuration
    """
    # check if it's already there:
    for channel in conda_config_as_dict()['channel URLs']:
        if USER in channel:
            print USER, "channel already there"
            break
    else:
        print "adding channel:", USER
        check_call(["conda", "config", "--add", "channels", "http://conda.binstar.org/%s"%USER])

def build_conda_skeleton(package):
    """
    Builds  conda package setup from pypi
    """
    print "Building package skeleton for:", package
    if os.path.os.path.isdir(package):
        print 'Skeleton for %s exists. delete it if you want it rebuilt'%package
    else:
        check_call(["conda", "skeleton", "pypi", package])
    
def build_package(package):
    """
    Builds the conda package
    """
    # check if it's already built
    package_file = check_output(["conda", "build", "--output", package]).strip()
    if os.path.exists(package_file):
        print "Package: %s\nalready built, not re-building"%package_file
    else:
        print "Building:", package
        check_call(["conda", "build", package])


def upload_package(package):
    """
    uploads the package to binstar
    """
    bld_dir = find_conda_build_dir()
    # find package filename:
    package_files = [name for name in os.listdir(bld_dir) if name.startswith(package)]
    package_files.sort()
    package_file = package_files[-1] # should get the latest version
    
    package_path = os.path.join(bld_dir, package_file)
    print package_path
    try:
        print "Uploading %s to binstar"%package
        DESCRIPTION = "Package: %s, auto-generated from PyPI"
        cmds = ["binstar", "upload",
                "-d", "DESCRIPTION",
                "--user", BINSTAR_USER,
                package_path]
        result = check_output(cmds,
                              stderr=STDOUT)
    except CalledProcessError as err:
        if "Distribution already exists" in err.output:
            print "Package: %s already is binstar. Delete it if you want the new copy uploaded"%package

def find_conda_build_dir():
    """
    returns the absolute path to the conda build directory
    """
    config = conda_config_as_dict()
    root = config['root environment'].split()[0]
    platform = config['platform']
    bld_dir = os.path.join(root, "conda-bld",platform)
    return bld_dir

def conda_config_as_dict():
    result = check_output(['conda', 'info'])
    result = result.split('\n')
    config = {}
    for i in range(1,len(result)):
        line = result[i].strip()
        if line:            
            key, val = [item.strip() for item in line.split(':', 1)]
            config[key] = val
            if key == 'channel URLs': #next lines may be more URLS
                config['channel URLs'] = [val]
                while True:
                    line = result[i+1].strip()
                    if line.startswith("http"):
                        config['channel URLs'].append(line)
                        i+=1
                    else:
                        break
    return config

if __name__ == "__main__":

    print "Adding binstar channels"
    add_binstar_channel("asmeurer")
    add_binstar_channel(BINSTAR_USER)

    try:
        package = sys.argv[1]
    except IndexError:
        print 'you must pass in a package name, or "all" if you want all the py_gnome dependencies built'
        sys.exit(1)

    if package == "all":
        packages = ['awesome-slugify',
                    'transaction',
                    'zope.sqlalchemy',
                    'iso8601',
                    'colander',
                    'geojson',
                    'waitress'
                    'webtest'
                    ]
    else:
        packages = [package]

    for package in packages:
        print
        print "Attempting to build/install:", package

        build_conda_skeleton(package)
        build_package(package)
        upload_package(package)



