""" 
Make scripts a package.
Include functions here that all scripts may want to use
"""
import os

def remove_netcdf(netcdf_file):
    """ 
    remove netcdf_file and associated uncertain netcdf file 

    Give scripts control over deleting the netcdf file before instantiating
    a new NetCDFOutput object.
    """
    if os.path.exists(netcdf_file):
        os.remove(netcdf_file)
        print "removed {0}".format(netcdf_file)
    
    file_, ext = os.path.splitext(netcdf_file)
    if os.path.exists(file_+'_uncertain'+ext):
        os.remove(file_+'_uncertain'+ext)
        print "removed {0}".format(netcdf_file)
    
