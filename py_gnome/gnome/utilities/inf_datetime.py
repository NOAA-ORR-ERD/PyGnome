#!/usr/bin/env python

"""
special datetime objects with -inf and inf times

These are not real datetime objects -- simply something that can be compared
with one

Note: this is very liberal with comparisons -- essentially an infinity
time object is greater than anything, so no checking to see what you are
comparing to..

Also special constructor for a real datetime that won't allow values out of
range for this application
"""

import datetime

## minimum and maximum valid datetime values

min_datetime = datetime.datetime(1970, 01, 01)
max_datetime = datetime.datetime(2038, 01, 19)


class InfTime(object):
    """
    class representing time into infinity

    compares as greater than any datetime (or any other object..)
    """

    def __str__(self):
        return 'Infinite time object'

    def __repr__(self):
        return ('{0.__class__.__module__}.{0.__class__.__name__}'
                '()'.format(self))

    def isoformat(self):
        return 'inf'

    def __lt__(self, other):
        'an InfTime object is never less than any other object'
        return False

    def __le__(self, other):
        """
            an InfTime object is never less than or equal to any
            other object otehr than itself.
        """

        if isinstance(other, InfTime):
            return True
        else:
            return False

    def __eq__(self, other):
        'an InfTime object is only equal to itself'
        if isinstance(other, InfTime):
            return True
        else:
            return False

    def __ne__(self, other):
        'an InfTime object is only equal to itself'
        if isinstance(other, InfTime):
            return False
        else:
            return True

    def __gt__(self, other):
        '''
           an InfTime object is greater than eveything except another
           InfTime object
        '''
        if isinstance(other, InfTime):
            return False
        else:
            return True

    def __ge__(self, other):
        'an InfTime object is greater or equal to anything'
        return True

    def __pos__(self):
        return self

    def __neg__(self):
        return MinusInfTime()


class MinusInfTime(object):
    """
    class representing time from infinity in the past

    compares as less than any datetime (or any other object)
    """
    def __str__(self):
        return 'Minus infinite time object'

    def __repr__(self):
        return ('{0.__class__.__module__}.{0.__class__.__name__}'
                '()'.format(self))

    def isoformat(self):
        return '-inf'

    def __lt__(self, other):
        '''
           an MinusInfTime object is always less than any other object
           except itself
        '''
        if isinstance(other, MinusInfTime):
            return False
        else:
            return True

    def __le__(self, other):
        '''
           an MinusInfTime object is always less than or equal to any
           other object.
        '''

        return True

    def __eq__(self, other):
        'A MinusInfTime object is only equal to itself'

        if isinstance(other, MinusInfTime):
            return True
        else:
            return False

    def __ne__(self, other):
        'A MinusInfTime object is only equal to itself'
        if isinstance(other, MinusInfTime):
            return False
        else:
            return True

    def __gt__(self, other):
        'A MinusInfTime object is not greater than anything'
        return False

    def __ge__(self, other):
        '''
           A Minus InfTime object is not greater than or equal
           to anything other than itself
        '''
        if isinstance(other, MinusInfTime):
            return True
        else:
            return False

    def __pos__(self):
        return self

    def __neg__(self):
        return InfTime()


class InfDateTime(datetime.datetime):
    """
    A special datetime object:

    It is either a regular datetime object, with the provisio
    that it can't be set outside the range given my the module
    variables:

    min_datetime
    max_datetime

    or a MinusInfTime or InfTime object.
    """
    def __new__(
        cls,
        year,
        month=None,
        day=None,
        hour=0,
        minute=0,
        second=0,
        microsecond=0,
        tzinfo=None,
        ):
        """
        create a new InfDateTime object

        :param year: integer year of datetime object.
                     year can also be "inf" or "-inf", and you'll get a special
                     MinusInfTime or InfTime object.

        :param month: integer month of datetime
        :param day: integer day of datetime
        :param hour=0: integer hour of datetime
        :param minute=0: integer minute of datetime
        :param second=0: integer second of datetime
        :param microsecond=0: integer microsecond of datetime
        :param tzinfo=None: timzoneinfo object.  See datetime docs for details.
        """

        # special values
        if year == 'inf':
            return InfTime()
        elif year == '-inf':
            return MinusInfTime()
        else:
            try:
                int(year)
            except ValueError:
                raise ValueError('Year must be an integer, or "inf" or "-inf"')

            if month is None:
                raise TypeError("Required argument 'month' (pos 2) not found")

            if day is None:
                raise TypeError("Required argument 'day' (pos 3) not found")

            inst = super(InfDateTime, cls).__new__(cls,
                                                   year, month, day,
                                                   hour, minute, second,
                                                   microsecond, tzinfo)

            if inst > max_datetime:
                raise ValueError("InfDateTime can't be created that is after:"
                                 " {0}".format(max_datetime.isoformat()))
            elif inst < min_datetime:
                raise ValueError("InfDateTime can't be created that is before:"
                                 " {0}".format(min_datetime.isoformat()))
            return inst


if __name__ == '__main__':
    dt = InfDateTime('inf')
    print dt, type(dt)

    dt = InfDateTime('-inf')
    print dt, type(dt)

    dt = InfDateTime(2012, 4, 20)
    print dt, type(dt)
