#!/usr/bin/env python

import numpy
np = numpy


class Time(object):
    @classmethod
    def MinutesToSeconds(cls, minutes=1.0):
        return minutes * 60

    @classmethod
    def HoursToSeconds(cls, hours=1.0):
        return hours * (cls.MinutesToSeconds() * 60)

    @classmethod
    def DaysToSeconds(cls, days=1.0):
        return days * (cls.HoursToSeconds() * 24)

    @classmethod
    def SecondsToDays(cls, seconds=1.0):
        return seconds / cls.DaysToSeconds()


class Length(object):
    @classmethod
    def MetersToInches(cls, length=1.0):
        '1 Meter = 39.3700787 Inches'
        return length * 39.3700787

    @classmethod
    def MetersToFeet(cls, length=1.0):
        return length * cls.MetersToInches(length) / 12


class Area(object):
    @classmethod
    def SquareMetersToSquareInches(cls, length=1.0):
        '1 m^2 = 1550 in^2'
        return length * (Length.MetersToInches(length) ** 2)


class Volume(object):
    @classmethod
    def CubicMetersToScf(cls, volume=1.0):
        '''
            Convert volume ratio units as (m^3) -> (standard cubic feet)
            1 m^3 == 35.3147 scf
        '''
        return volume * (Length.MetersToFeet(length=1.0) ** 3)

    @classmethod
    def CubicMetersToStb(cls, volume=1.0):
        '''
            Convert volume ratio units as (m^3) -> (stock tank barrel)
            1 m^3 == 6.289808 stb
        '''
        return volume * 6.289808

    @classmethod
    def CubicMeterRatioToScfPerStb(cls, volumeRatio=1.0):
        '''
            Convert volume ratio units as:
                (S m^3/ S m^3) -> (standard cubic feet/ stock tank barrel)
            TODO: This should probably be moved to a common location
                  like hazpy.unit_conversion
        '''
        return volumeRatio * (cls.CubicMetersToScf() / cls.CubicMetersToStb())

    @classmethod
    def ScfPerStbToCubicMeterRatio(cls, volumeRatio=1.0):
        '''
            Convert volume ratio units as:
                (standard cubic feet/ stock tank barrel) -> (S m^3/ S m^3)
            TODO: This should probably be moved to a common location
                  like hazpy.unit_conversion
        '''
        return volumeRatio * (cls.CubicMetersToStb() / cls.CubicMetersToScf())

    @classmethod
    def CubicMetersPerSecondToStbsPerDay(cls, volumeRate=1.0):
        return Time.DaysToSeconds(cls.CubicMetersToStb(volumeRate))


class Force(object):
    '''
        TODO: This should probably be moved to a common location
              like hazpy.unit_conversion
    '''
    gravity = 9.80665  # m/s^2

    @classmethod
    def NewtonsToPounds(cls, force=1.0):
        '''
            Convert volume ratio units as (m^3) -> (stock tank barrel)
            1 Newton = 0.224808943 pounds force
        '''
        return force * 0.224808943

    @classmethod
    def PascalsToPsi(cls, pressure=1.0):
        '''
            Pascal = Newton / m^2
            PSI    = lbs / in^2
        '''
        return pressure * (cls.NewtonsToPounds() / Area.SquareMetersToSquareInches())

    @classmethod
    def PsiToPascals(cls, pressure=1.0):
        '''
            Pascal = Newton / m^2
            PSI    = lbs / in^2
        '''
        return pressure / cls.PascalsToPsi()


if __name__ == '__main__':

    assert np.isclose(Time.SecondsToDays(Time.DaysToSeconds()), 1.0)

    volumeRatio = 2530

    assert np.isclose(Force.PascalsToPsi(), 1.450377e-4)
    assert np.isclose(Force.PsiToPascals(), 6.8948e3)

    assert np.isclose(Volume.CubicMetersToStb(), 6.289808)
    assert np.isclose(Volume.CubicMetersToScf(), 35.3147)

    print '%s / %s == %s' % (Volume.CubicMetersToScf(),
                             Volume.CubicMetersToStb(),
                             Volume.CubicMeterRatioToScfPerStb())
    print volumeRatio, Volume.CubicMeterRatioToScfPerStb(volumeRatio)

    assert np.isclose(Volume.CubicMeterRatioToScfPerStb(), 5.61458578917)

    # the conversions for m^3/m^3 <--> scf/stb should be symmetric
    assert np.isclose(Volume.CubicMeterRatioToScfPerStb(Volume.ScfPerStbToCubicMeterRatio()), 1.0)

    assert np.isclose(Volume.CubicMetersPerSecondToStbsPerDay(), 543439.4112)
