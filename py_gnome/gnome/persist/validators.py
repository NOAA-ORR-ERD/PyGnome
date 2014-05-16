'''
Created on Feb 26, 2013

Define general purpose functions that can used as validators
'''
import time

import numpy
np = numpy

from colander import Invalid

from gnome.utilities.inf_datetime import InfTime, MinusInfTime


def positive(node, value):
    if value <= 0:
        raise Invalid(node, 'Value must be greater than zero.')


def convertible_to_seconds(node, value):
    """ validate only datetime objects """
    if isinstance(value, MinusInfTime) or isinstance(value, InfTime):
        return

    try:
        time.mktime(list(value.timetuple()))
    except (OverflowError, ValueError):
        raise Invalid(node, 'Invalid date.')


def no_duplicate_datetime(node, values):
    """
    Check for duplicate datetime values in numpy structured array like
    datetime_value_2d
    Reject ``values`` if it contains duplicates.
    """
    try:
        unique = np.unique(values['time'])
    except AttributeError:
        return

    num_dups = len(values) - len(unique)

    if num_dups:
        raise Invalid(node,
                      'Duplicate time entries are not allowed. '
                      'Found {0} duplicates.'.format(num_dups))


def ascending_datetime(node, values):
    """
    Check the datetime values in numpy structured array
    (like datetime_value_2d)
    are in ascending order
    """
    # check to make sure the time values are in ascending order
    if np.any(values['time'][np.argsort(values['time'])]
              != values['time']):
        raise Invalid(node,
                      'The datetime values in the timeseries must be in '
                      'ascending order')


#==============================================================================
# def degrees_true(node, direction):
#    if 0 > direction > 360:
#        raise Invalid(
#            node, 'Direction in degrees true must be between 0 and 360.')
#
#
# def get_direction_degree(direction):
#   """
#   Convert user input for direction into degrees.
#   """
#   if direction.isalpha():
#       return util.DirectionConverter.get_degree(direction)
#   else:
#       return direction
#
# def cardinal_direction(node, direction):
#   if not util.DirectionConverter.is_cardinal_direction(direction):
#       raise Invalid(
#           node, 'A cardinal directions must be one of: %s' % ', '.join(
#               util.DirectionConverter.DIRECTIONS))
#
# def valid_direction(node, value):
#    """
#    Unused.
#    """
#    try:
#        degrees_true(node, float(value))
#    except ValueError:
#        cardinal_direction(node, value.upper())
#==============================================================================
