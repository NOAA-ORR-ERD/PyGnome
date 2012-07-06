# -*- coding: utf-8 -*-
"""
@author: brian.zelenke
"""

import sys
import os
import netCDF4
import numpy as np


def get_tests(TestsFolder):
    """
    Return all the folders in the specified directory whose file-names start
    with "Test_".
    """
    folders=os.listdir(TestsFolder)
    folders=[name for name in folders if name.startswith("Test_")]
    return folders
    
def run_gnome(test,gnome_exe):
    """
    Specify a directory ("test") and the path to the GNOME executable
    (gnome_exe) to run the command.txt file within the "test" directory.
    """
    command_file=os.path.join(test,"command.txt")
    cmd="%s %s"%(gnome_exe,command_file)
    os.system(cmd)

if __name__=="__main__":
    """
    If you call this script with your present working directory in the same
    folder as this script itself, run each command.txt file in each Test_*
    sub-folder using the copy of Gnome.exe at the path hard-coded below.
    """
    gnome_exe="gnome\\Gnome.exe"
    folders=get_tests("./")
    loop=-1
    for test in folders:
        loop=loop+1
        print "Running test: ",test
        run_gnome(test,gnome_exe)
        
        known_file="%s_test_reference.nc"%folders[loop][-2:]
        known_file=netCDF4.Dataset(os.path.join(test,known_file))
        known_vars=known_file.variables.keys()
        test_file="%s_test_out.nc"%folders[loop][-2:]
        test_file=netCDF4.Dataset(os.path.join(test,test_file))
        test_vars=test_file.variables.keys()
        
        if known_vars==test_vars:
            print "Variable names in the NetCDF file match those known: ",test
        else:
            sys.stderr.write("Error:  Variable names in the NetCDF file do not match those known: ",test)
        
        for varname in known_vars:
            if np.allclose(known_file.variables[varname],test_file.variables[varname]):
                print "%s: Variable %s matches known-good copy." %(test,varname)
            else:
                sys.stderr.write("%s: Variable %s does not match known-good copy." %(test,varname))