"""
utilities for scripting  gnome

assorted utilities that make it easier to write scripts to automate gnome

designed to be imported into the package __init__.py

remember to add anyting new you want imported to "__all__"

"""
import os
import shutil

import gnome


def make_images_dir(images_dir=None):
    """
    Create output directory for rendered images.
    If it already exists, delete all old output files

    """
    if images_dir is None:
        images_dir = os.path.join(os.getcwd(), 'images')

    print 'images_dir is:', images_dir

    if os.path.isdir(images_dir):
        print 'removing...', images_dir
        shutil.rmtree(images_dir)
    os.mkdir(images_dir)


def remove_netcdf(netcdf_file):
    """
    remove netcdf_file and associated uncertain netcdf file

    Give scripts control over deleting the netcdf file before instantiating
    a new NetCDFOutput object.
    """
    if os.path.exists(netcdf_file):
        os.remove(netcdf_file)
        print 'removed {0}'.format(netcdf_file)

    (file_, ext) = os.path.splitext(netcdf_file)
    if os.path.exists(file_ + '_uncertain' + ext):
        os.remove(file_ + '_uncertain' + ext)
        print 'removed {0}'.format(netcdf_file)


def set_verbose(log_level='info'):
    """
    Set the logging system to dump to the console --
    you can see much more what's going on with the model
    as it runs

    :param log_level='info': the level you want your log to show. options are,
                             in order of importance: "debug", "info", "warning",
                             "error", "critical".

    You will only get the logging messages at or above the level you set.
    Set to "debug" for everything.
    """
    gnome.initialize_console_log(log_level)
