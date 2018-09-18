"""
time_utils.py

Handy utilitties ofr working with time

mostly easy ways to get the right units with a timedelta object

"""

__all__ = ['seconds',
           'hours',
           'minutes',
           'days',
           'weeks',
           ]


from datetime import timedelta


def seconds(seconds=1):
    return timedelta(seconds=seconds)


def minutes(minutes=1):
    return timedelta(minutes=minutes)


def hours(hours=1):
    return timedelta(hours=hours)


def days(days=1):
    return timedelta(days=days)


def weeks(weeks=1):
    return timedelta(days=weeks * 7)
