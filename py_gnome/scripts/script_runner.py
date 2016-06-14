#!/usr/bin/env python

'''
Created on Apr 4, 2013

Command line utility to run a gnome script
'''

import os
import shutil
import argparse
import sys
import imp

from gnome import scripting
from gnome.persist import load


def run(model):

    # create a place for test images (cleaning out any old ones)
    # images_dir = os.path.join( base_dir, "images")
    # if os.path.isdir(images_dir):
    #    shutil.rmtree(images_dir)
    # os.mkdir(images_dir)
    #

    print 'running:'

    # run the model

    while True:
        print 'calling step'
        try:
            image_info = model.step()
            print image_info
        except StopIteration:
            print 'Done with the model run'
            break


def save(model, saveloc):

    # save directory
    # saveloc = os.path.join(base_dir, "save_model")

    if os.path.isdir(saveloc):
        shutil.rmtree(saveloc)
    os.mkdir(saveloc)
    print 'saving ..'
    model.save(saveloc)


def run_from_save(saveloc_model):
    if not os.path.isfile(saveloc_model):
        raise ValueError('{0} does not appear to be a valid'
                         ' json file'.format(saveloc_model))
    model = load(saveloc_model)

    model.rewind()

    run(model)


def parse_args(argv):
    """parse command line arguments and process"""

    parser = argparse.ArgumentParser()
    parser.add_argument('location',
                        help=('path to script to run or save, or save file'
                              ' location to be loaded and run'), nargs=1)
    parser.add_argument('--do', default='run',
                        help='either run or save the model',
                        choices=['run', 'save', 'run_from_save'])

    parser.add_argument('--saveloc',
                        help=('store save files here. Defaults to'
                              ' location/save_model'), nargs=1)
    parser.add_argument('--images',
                        help=('store output images here. Defaults to'
                              ' location/images'), nargs=1)
    args = parser.parse_args(argv)

    args.location = args.location[0]

    if not os.path.exists(args.location):
        raise IOError('The path {0} does not exist'.format(args.location))

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


def load_model(location, images_dir):

    # import ipdb; ipdb.set_trace()

    filename = os.path.split(location)[1]

    scripting.make_images_dir(images_dir)
    imp_script = imp.load_source(filename.rstrip('.py'), location)
    model = imp_script.make_model(images_dir)
    return (model, imp_script)


if __name__ == '__main__':
    args = parse_args(sys.argv[1:])

    if args.do == 'run' or args.do == 'save':
        if not os.path.isfile(args.location):
            raise ValueError("{0} is not a file - provide a python script if"
                   " action is to 'run' or 'save' model".format(args.location))

        (model, imp_script) = load_model(args.location, args.images)

    if args.do == 'run':
        run(model)
        try:
            imp_script.post_run(model)
        except AttributeError:
            # must not have a post_run function
            pass

    elif args.do == 'save':
        save(model, args.saveloc)
    else:

            # if args.do == 'run_from_save':

        run_from_save(args.saveloc)

    # if args.do in ('run','run_from_save'):
    #    post_run(model, imp_script)    # cannot do post run because it needs module name to call post_run
