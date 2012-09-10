#!/usr/bin/env python

"""
Assorted utilities for manipulating latitude and longitude values

"""

__version__ = "1.3"

import math, struct


def signbit(value):
    """
    Test whether the sign bit of the given floating-point value is
    set.  If it is set, this generally means the given value is
    negative.  However, this is not the same as comparing the value
    to C{0.0}.  For example:

    >>> NEGATIVE_ZERO < 0.0
    False

    since negative zero is numerically equal to positive zero.  But
    the sign bit of negative zero is indeed set:

    >>> signbit(NEGATIVE_ZERO)
    True
    >>> signbit(0.0)
    False

    @type  value: float
    @param value: a Python (double-precision) float value

    @rtype:  bool
    @return: C{True} if the sign bit of C{value} is set;
             C{False} if it is not set.
    
    signbit and doubleToRawLongBits
    are from Martin Jansche:

    http://symptotic.com/mj/code.html  (MIT license).

    This is required to capture the difference between -0.0 and 0.0, which is
    useful if someone wants to convert a latitude or longitude like:
    -0.0degrees, 34minutes to  0d34'00"S

    """
    return (doubleToRawLongBits(value) >> 63) == 1

def doubleToRawLongBits(value):
    """
    @type  value: float
    @param value: a Python (double-precision) float value

    @rtype: long
    @return: the IEEE 754 bit representation (64 bits as a long integer)
             of the given double-precision floating-point value.
    """
    # pack double into 64 bits, then unpack as long int
    return struct.unpack('Q', struct.pack('d', value))[0]

class LatLongConverter:
    @classmethod
    def ToDecDeg(self, d=0, m=0, s=0, ustring = False, max=180):
        """
        DecDegrees = ToDecDeg(d=0, m=0, s=0)
        
        converts degrees, minutes, seconds to decimal degrees (returned as a Float).
        """
        if  m < 0 or s < 0:
            raise ValueError("Minutes and Seconds have to be positive")
        if m > 60.0 or s > 60.0:
            raise ValueError("Minutes and Seconds have to be between -180 and 180")
        if abs(d) > max:
            raise ValueError("Degrees have to be between -180 and 180")

        if signbit(d):
            Sign = -1
            d = abs(d)
        else:
            Sign = 1
        
        deg_has_fract = bool(math.modf(d)[0])
        min_has_fract = bool(math.modf(m)[0])
        if deg_has_fract and (m != 0.0 or s != 0.0):
            raise ValueError("degrees cannot have fraction unless both minutes"
                             "and seconds are zero")
        if min_has_fract and s != 0.0:
            raise ValueError("minutes cannot have fraction unless seconds are zero")

        DecDegrees = Sign * (d + m/60.0 + s/3600.0)

        if ustring:
            return u"%.6f\xb0"%(DecDegrees)
        else:
            return DecDegrees

    @classmethod
    def ToDegMin(self, DecDegrees, ustring = False):
        """
        Converts from decimal (binary float) degrees to:
          Degrees, Minutes

        If the optional parameter: "ustring" is True,
        a Unicode string is returned

        """
        if signbit(DecDegrees):
            Sign = -1
            DecDegrees = abs(DecDegrees)
        else:
            Sign = 1
        Degrees = int(DecDegrees)
        DecMinutes = round((DecDegrees - Degrees + 1e-14) * 60, 10)# add a tiny bit then round to avoid binary rounding issues
        if ustring:
            if Sign == 1:
                return u"%i\xb0 %.3f'"%(Degrees, DecMinutes)
            else:
                return u"-%i\xb0 %.3f'"%(Degrees, DecMinutes)
        else:
            return (Sign*float(Degrees), DecMinutes) # float to preserve -0.0

    @classmethod
    def ToDegMinSec(self, DecDegrees, ustring = False):

        """
        Converts from decimal (binary float) degrees to:
          Degrees, Minutes, Seconds

        If the optional parameter: "ustring" is True,
        a unicode string is returned

        """
        if signbit(DecDegrees):
            Sign = -1
            DecDegrees = abs(DecDegrees)
        else:
            Sign = 1
        Degrees = int(DecDegrees)
        DecMinutes = (DecDegrees - Degrees + 1e-14) * 60 # add a tiny bit to avoid rounding issues

        Minutes = int(DecMinutes)
        Seconds = round(((DecMinutes - Minutes) * 60), 10 )
        if ustring:
            if Sign == 1:
                return u"%i\xb0 %i' %.2f\""%(Degrees, Minutes, Seconds)
            else:
                return u"-%i\xb0 %i' %.2f\""%(Degrees, Minutes, Seconds)
        else:
            return (Sign * float(Degrees), Minutes, Seconds)

## These are classes used in our web apps: ResponseLink, etc.
## They provide a different interface to lat-long format conversion
class Latitude:
    """An object that can interpret a latitude in various formats.

       Constructor:
       Latitude(deg, min=0.0, sec=0.0, direction=None)
           - 'deg' may be between -90.0 and 90.0.
           - if 'min' is nonzero, 'deg' cannot have a fractional part.
             (This means 5 and 5.0 are acceptable but 5.1 is not.)
           - if 'sec' is nonzero, 'deg' and 'min' cannot have fractional parts.
           - 'direction' may be a string beginning with 'N' or 'S' (case
             insensitive), or None.
           - if 'direction' is not None, 'deg' cannot be negative.

       Attributes:
       .value : a float in decimal degrees.  Positive is North; negative is
           South.  (These apply to zero too; positive zero is North.)

       Methods:
       .degrees() -> (float, str)            
       .degrees_minutes() -> (int, float, str)
       .degrees_minutes_seconds() -> (int, int, float, str)
       The 'str' argument is the direction: "North" or "South".

       Example:
       >>> lat1 = Latitude(-120.7625)
       >>> lat2 = Latitude(-120, 45.7500)
       >>> lat3 = Latitude(-120, 45, 45)
       >>> lat4 = Latitude(120.7625, direction='South')
       >>> lat5 = Latitude(120, 45.7500, direction='S')
       >>> lat6 = Latitude(120, 45, 45, direction='south')
       >>> (lat1.value == lat2.value == lat3.value == lat4.value ==
       ... lat5.value == lat6.value)
       True
       >>> lat1.value
       -120.7625
       >>> lat1.degrees()
       (120.7625, 'South')
       >>> lat1.degrees_minutes()
       (120, 45.750000000000171, 'South')
       >>> lat1.degrees_minutes_seconds()
       (120, 45, 45.000000000010232, 'South')
       >>> print str(lat1)
       Latitude(-120.762500)
    """
    negative_direction = "South"
    positive_direction = "North"
    min = -90.0
    max = 90.0

    def __init__(self, deg, min=0.0, sec=0.0, direction=None):
        ndir = self.negative_direction[0].upper()
        pdir = self.positive_direction[0].upper()

        if direction:
            if deg < 0.0:
                msg = "degrees cannot be negative if direction is specified"
                raise ValueError(msg)
            if   direction[0].upper() == pdir:
                pass
            elif direction[0].upper() == ndir:
                deg = -deg
            else:
                msg = "direction must start with %r or %r" % (pdir, ndir)
                raise ValueError(msg)
        
        self.value = LatLongConverter.ToDecDeg(deg, min, sec, max=self.max)
        
    def direction(self):
        if self.value < 0.0:
            return self.negative_direction
        else:
            return self.positive_direction

    def degrees(self):
        deg = abs(self.value)
        return deg, self.direction()

    def degrees_minutes(self):
        deg, min = LatLongConverter.ToDegMin(abs(self.value))
        return deg, min, self.direction()

    def degrees_minutes_seconds(self):
        deg, min, sec = LatLongConverter.ToDegMinSec(abs(self.value))
        return deg, min, sec, self.direction()

    def __repr__(self):
        try:
            return "%s(%f)" % (self.__class__.__name__, self.value)
        except AttributeError:
            return "%s(uninitialized)" % self.__class__.__name__

    def format(self, style):
        """
        format(style)
        
        returns formatted value as Unicode string with u'\xb0' (degree symbol).
        
        style is one of:
        1:  decimal degrees
        2:  degrees, decimal minutes
        3:  degrees, minutes, seconds
        """
        
        if   style == 1:
            return u'''%0.2f\xb0 %s''' % self.degrees()
        elif style == 2:
            return u'''%d\xb0 %0.2f' %s''' % self.degrees_minutes()
        elif style == 3:
            return u'''%d\xb0 %d' %0.2f" %s''' % self.degrees_minutes_seconds()
        else:
            raise ValueError("style must be 1, 2, or 3")

    def format_html(self, style):
        """
        format_html(style)
        
        Backward compatibility for Quixote rlink and Pylons inews.
        """
        return self.format(style).replace(u"\xb0", u"&deg;").encode("ascii")

class Longitude(Latitude):
    """See Latitude docstring.
    
       Positive is East; negative is West.  Degrees must be between -180.0 and
       180.0
    """
    negative_direction = "West"
    positive_direction = "East"
    min = -180.0
    max = 180.0
    
class DummyLatitude:
    """A pseudo-Latitude whose components are None.
       Useful in building HTML forms where the value is not required.

       Note: this class may be deleted if it doesn't turn out to be useful.
    """
    value = None
    def direction(self):                return None
    def degrees(self):                  return None, None
    def degrees_minutes(self):          return None, None, None
    def degrees_minutes_seconds(self):  return None, None, None, None

class DummyLongitude(DummyLatitude):
    """
       Note: this class may be deleted if it doesn't turn out to be useful.
    """
    pass

