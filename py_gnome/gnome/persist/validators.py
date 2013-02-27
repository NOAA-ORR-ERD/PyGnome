'''
Created on Feb 26, 2013

Define general purpose functions that can used as validators
'''
import datetime,time

import numpy 

import gnome.basic_types
from colander import (
                      Invalid,
                      OneOf,
                      )
#from webgnome import util
#from webgnome.model_manager import WebWind


#===============================================================================
# def get_direction_degree(direction):
#    """
#    Convert user input for direction into degrees.
#    """
#    if direction.isalpha():
#        return util.DirectionConverter.get_degree(direction)
#    else:
#        return direction
# 
# def cardinal_direction(node, direction):
#    if not util.DirectionConverter.is_cardinal_direction(direction):
#        raise Invalid(
#            node, 'A cardinal directions must be one of: %s' % ', '.join(
#                util.DirectionConverter.DIRECTIONS))
#===============================================================================


def positive(node, value):
    if value <= 0:
        raise Invalid(node, 'Value must be greater than zero.')


def zero_or_greater(node, value):
    if value < 0:
        raise Invalid(node, 'Value must be zero or greater.')


def convertable_to_seconds(node, value):
    try:
        time.mktime(list(value.timetuple()))
    except (OverflowError, ValueError) as e:
        raise Invalid(node, 'Invalid date.')


def degrees_true(node, direction):
    if 0 > direction > 360:
        raise Invalid(
            node, 'Direction in degrees true must be between 0 and 360.')


def no_duplicates(node, values):
    """
    Reject ``values`` if it contains duplicates.
    """
    try:
        unique = numpy.unique(values)
    except AttributeError:
        return

    num_dups = len(values) - len(unique)

    if num_dups:
        raise Invalid(
            node, 'Duplicates are not allowed. Found %s duplicates.' % num_dups)


def valid_direction(node, value):
    """
    Unused.
    """
    try:
        degrees_true(node, float(value))
    except ValueError:
        cardinal_direction(node, value.upper())
