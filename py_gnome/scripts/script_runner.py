#!/usr/bin/env python
'''
Created on Apr 4, 2013

Command line utility to run a gnome script
'''

import os
import shutil
from datetime import datetime, timedelta
import argparse
import sys
import imp

import numpy as np

import gnome
from gnome.environment import Wind, Tide
from gnome.utilities import map_canvas
from gnome.utilities.file_tools import haz_files
from gnome.persist import scenario


def make_model():
    
    return model

def run(model, images_dir):
    
    # create a place for test images (cleaning out any old ones)
    #images_dir = os.path.join( base_dir, "images")
    if os.path.isdir(images_dir):
        shutil.rmtree(images_dir)
    os.mkdir(images_dir)
    
    print "running:"
    
    # run the model
    while True:
        print "calling next_image"
        try:
            image_info = model.next_image(images_dir)
            print image_info
        except StopIteration:
            print "Done with the model run"
            break

def save(model, saveloc):
    # save directory
    #saveloc = os.path.join(base_dir, "save_model")
    if os.path.isdir(saveloc):
        shutil.rmtree(saveloc)
    os.mkdir(saveloc)
    print "saving .."
    scenario.save(model,saveloc)

def run_from_save(saveloc, images_dir):
    if not os.path.isdir(saveloc):
        raise ValueError("{0} does not appear to be a valid directory".format(saveloc))
    model = scenario.load(saveloc)
    run( model, images_dir)
    

def parse_args(argv):
    """parse command line arguments and process"""
    parser = argparse.ArgumentParser()
    parser.add_argument('location', help='path to script to run or save, or save file location to be loaded and run', nargs=1)
    parser.add_argument('--do', default='run', help='either run or save the model', choices=['run','save','run_from_save'])
    
    parser.add_argument('--saveloc',help='store save files here. Defaults to location/save_model', nargs=1)
    parser.add_argument('--images', help='store output images here. Defaults to location/images', nargs=1)
    args = parser.parse_args(argv)
    
    args.location = args.location[0]
    
    if not os.path.exists( args.location):
        raise IOError("The path {0} does not exist".format(args.location))
    
    # define defaults here
    base_dir = os.path.dirname(args.location)
    if args.do == 'run_from_save':
        args.saveloc = args.location
        
    elif args.do == 'save':
        if args.saveloc is None:
            args.saveloc = os.path.join(base_dir, 'save_model')
        
    if args.images is None:
        args.images = os.path.join(base_dir, 'images')
    
    return args

if __name__=="__main__":
    """when script is run from command like, then call run()"""
    args = parse_args(sys.argv[1:])
    
    if args.do == 'run' or args.do == 'save':
        if not os.path.isfile(args.location):
            raise ValueError("{0} is not a file - provide a python script if action is to 'run' or 'save' model".format(args.location))
        
        #import ipdb; ipdb.set_trace()
        dir_name, filename = os.path.split(args.location)
        myscript = imp.load_source(filename.rstrip('.py'),args.location) 
        model = myscript.make_model()
        
        if args.do == 'run':
            run(model, args.images)
        else:
            save(model, args.saveloc)
    
    elif args.do == 'run_from_save':
        run_from_save(args.saveloc, args.images)