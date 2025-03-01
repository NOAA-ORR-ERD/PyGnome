"""

Handy utilities for working with time

Internally, py_Gnome uses ``datetime.timedelta`` objects to represent time spans.
But it is a bit awkward to create these objects::

    datetime.timedelta(seconds=3600)

The time_utils module provides handy utilities to make it easier to construct
these objects.

Examples:

``hours()`` -- represents one hour

``hours(12)`` -- represents 12 hours

As these functions return timedelta objects, you can do math with them::

  seconds() * 60
  days(2) + hours(12)

etc...

These are the full set:

| ``seconds``
| ``minutes``
| ``hours``
| ``days``
| ``weeks``

"""

from datetime import datetime, timedelta

now = datetime.now


def seconds(seconds=1):
    """
    returns a datetime.timedelta object representing the specified number of seconds

    :param seconds: number of seconds -- defaults to 1 second.
    :type seconds: int or float

    """
    return timedelta(seconds=seconds)


def minutes(minutes=1):
    """
    returns a datetime.timedelta object representing the specified number of minutes

    :param minutes: number of minutes -- defaults to 1 day.
    :type minutes: int or float
    """
    return timedelta(minutes=minutes)


def hours(hours=1):
    """
    returns a datetime.timedelta object representing the specified number of hours

    :param hours: number of hours -- defaults to 1 hour.
    :type hours: int or float
    """
    return timedelta(hours=hours)


def days(days=1):
    """
    returns a datetime.timedelta object representing the specified number of days

    :param days: number of days -- defaults to 1 day.
    :type days: int or float
    """
    return timedelta(days=days)


def weeks(weeks=1):
    """
    Returns a datetime.timedelta object representing the specified number of weeks

    :param weeks: number of weeks -- defaults to 1 week.
    :type weeks: int or float
    """
    return timedelta(days=weeks * 7)
