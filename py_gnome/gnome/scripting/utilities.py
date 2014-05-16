"""
utilities for scripting  gnome

assorted utilities that make it easier to write scripts to automate gnome

designed to be imported into the pacakge __init__.py

"""
import os
import shutil


def make_images_dir(images_dir=None):

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
