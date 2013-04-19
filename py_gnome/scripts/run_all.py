#!/usr/bin/env python

'''
Created on Apr 4, 2013

runs all scripts in here
'''

import os

scripts = ['script_boston/script_boston.py',
           'script_guam/script_guam.py',
           'script_chesapeake_bay/script_chesapeake_bay.py',
           'script_long_island/script_long_island.py',
           'script_mississippi_river/script_lower_mississippi.py'
           ]

for script in scripts:
    torun = "./script_runner.py "+script 
    os.system(torun)
    
    print "\n completed : {0}\n".format(torun)
    
    tosave = "./script_runner.py --do=save "+script
    os.system(tosave)
    
    print "\n completed : {0}\n".format(tosave)
    
    run_from_save= "./script_runner.py     --do=run_from_save "+os.path.join(os.path.dirname(script),'save_model')
    os.system(run_from_save)
    
    print "\n completed : {0}\n".format(run_from_save)
