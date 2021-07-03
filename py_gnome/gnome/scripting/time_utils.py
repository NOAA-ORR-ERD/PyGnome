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
    returns a datetime.timedelta object representing the specified number of seconds"

    :param seconds=1:
    """
    return timedelta(seconds=seconds)


def minutes(minutes=1):
    """
    returns a datetime.timedelta object representing the specified number of minutes

    :param minutes=1:
    """
    return timedelta(minutes=minutes)


def hours(hours=1):
    """
    returns a datetime.timedelta object representing the specified number of hours

    :param hours=1:
    """
    return timedelta(hours=hours)


def days(days=1):
    """
    returns a datetime.timedelta object representing the specified number of hours"

    :param hours=1:
    """
    return timedelta(days=days)


def weeks(weeks=1):
    """
    returns a datetime.timedelta object representing the specified number of weeks"

    :param weeks=1:
    """
    return timedelta(days=weeks * 7)
