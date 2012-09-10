#!/usr/bin/env python

"""Unit conversion calculators.

CHANGELOG:
  2005/07/14 CB: Initial import.
  2005/07/15 MO: Latitude and Longitude classes.
  2005/07/18 MO: Limit Latitude range to 90.
  2007/05/03 CB: Tweaked lat-long to get correct answer for 40 minutes.
                 Added "convert" as alias for "Convert"
  2007/12/17 MO: Add .format() method to Latitude/Longitude returning Unicode.
                 .factor_html() remains for backward compatibility, using
                 .format() internally.
                 Make Latitude/Longitude.__repr__ more robust in case
                 .__init__ raises an exception; workaround for Pylons bug
                 http://pylonshq.com/project/pylonshq/ticket/341
  2008/02/22 CB: Added a few more units for Ian
  2008/06/05 CB: Various changes before putting the Converter GUI on the web: new units, changed, spelling, etc.
  2009/09/29 CB: Re-factored the lat-long stuff:
                 - it's not in a separate module
                 - Mike and Chris' code has been merged for less duplication
                 - Unit data moved to separate module
"""

__version__ = "1.2.2"

from unit_data import ConvertDataUnits
from lat_long import LatLongConverter, Latitude, Longitude, DummyLatitude, DummyLongitude # for backward compatibility

## A few utilities
def Simplify(String):
    """
    Simplify(String)

    returns the string with the whitespace and capitalization removed
    """
    return "".join(String.lower().split())

def GetUnitTypes():
    return ConvertDataUnits.keys()

def GetUnitNames(UnitType):
    return ConvertDataUnits[UnitType].keys()

class ConverterClass:
    def __init__(self, TypeName, UnitsDict):
        self.Name = TypeName

        self.Synonyms = {}
        self.Convertdata = {}
        for PrimaryName, data in UnitsDict.items():
            # strip out whitespace and capitalization
            Pname = Simplify(PrimaryName)
            self.Convertdata[Pname] = data[0]
            self.Synonyms[Pname] = Pname
            for synonym in data[1]:
                self.Synonyms[Simplify(synonym)] = Pname

    def Convert(self, FromUnit, ToUnit, Value):

        """
        Convert(FromUnit, ToUnit, Value)

        returns a new value, in the units of ToUnit.

        """
        FromUnit = Simplify(FromUnit)
        ToUnit = Simplify(ToUnit)

        try:
            FromUnit = self.Synonyms[FromUnit]
        except KeyError:
            raise InvalidUnitError(FromUnit, self.Name)
        try:
            ToUnit = self.Synonyms[ToUnit]
        except KeyError:
            raise InvalidUnitError(ToUnit, self.Name)

        return Value * self.Convertdata[FromUnit] / self.Convertdata[ToUnit]

# the special case classes:
class TempConverterClass(ConverterClass):
    def Convert(self, FromUnit, ToUnit, Value):

        """
        Convert(FromUnit, ToUnit, Value)

        returns a new value, in the units of ToUnit.

        """

        FromUnit = Simplify(FromUnit)
        ToUnit = Simplify(ToUnit)

        try:
            FromUnit = self.Synonyms[FromUnit]
        except KeyError:
            raise InvalidUnitError(FromUnit, self.Name)
        try:
            ToUnit = self.Synonyms[ToUnit]
        except KeyError:
            raise InvalidUnitError(ToUnit, self.Name)

        A1 = self.Convertdata[FromUnit][0]
        B1 = self.Convertdata[FromUnit][1]
        A2 = self.Convertdata[ToUnit][0]
        B2 = self.Convertdata[ToUnit][1]

        #to_val = (round((from_val * A1 + B1),13) - round(B2,13))*A2 # rounding to get rid of cancelation error
        to_val = ((Value + B1)*A1/A2)-B2
        return to_val

class DensityConverterClass(ConverterClass):
    def Convert(self, FromUnit, ToUnit, Value):

        """
        Convert(FromUnit, ToUnit, Value)

        returns a new value, in the units of ToUnit.

        """

        FromUnit = Simplify(FromUnit)
        ToUnit = Simplify(ToUnit)
         
        try:
            FromUnit = self.Synonyms[FromUnit]
        except KeyError:
            raise InvalidUnitError(FromUnit, self.Name)
        try:
            ToUnit = self.Synonyms[ToUnit]
        except KeyError:
            raise InvalidUnitError(ToUnit, self.Name)
        if FromUnit == "apidegree": # another Special case (could I do this the same as temp?)
            Value = 141.5/(Value + 131.5)
            FromUnit = u"specificgravity(15\xb0c)"
        if ToUnit == "apidegree":
            ToVal = 141.5/(Value * self.Convertdata[FromUnit] / self.Convertdata[u"specificgravity(15\xb0c)"] ) - 131.5
        else:
            ToVal = Value * self.Convertdata[FromUnit] / self.Convertdata[ToUnit]
        return ToVal

class OilQuantityConverter:
    @classmethod
    def ToVolume(self, Mass, MassUnits, Density, DensityUnits, VolumeUnits):
        Density = convert("Density", DensityUnits, "kg/m^3", Density)
        #print "Density in kg/m^3", Density
        Mass = convert("Mass", MassUnits, "kg", Mass)
        #print "Mass in kg", Mass
        Volume = Mass / Density
        #print "Volume in m^3", Volume
        Volume = convert("Volume", "m^3", VolumeUnits, Volume)
        #print "Volume in %s"%VolumeUnits, Volume
        return Volume
    @classmethod
    def ToMass(self, Volume, VolUnits, Density, DensityUnits, MassUnits):
        Density = convert("Density", DensityUnits, "kg/m^3", Density)
        #print "Density in kg/m^3", Density
        Volume = convert("Volume", VolUnits, "m^3", Volume)
        #print "Volume in m^3", Volume
        Mass = Volume * Density
        #print "Mass in kg", Mass
        Mass = convert("Mass", "kg", MassUnits,  Mass)
        #print "Mass in %s"%MassUnits, Mass
        return Mass

# create the converter objects
Converters = {}
for (unittype,data) in ConvertDataUnits.items():
    if unittype.lower() == 'temperature':
        Converters["temperature"] = TempConverterClass(unittype, data)
    elif unittype.lower() == 'density':
        Converters["density"] = DensityConverterClass(unittype, data)
    else:
        Converters[Simplify(unittype)] = ConverterClass(unittype, data)

def convert(UnitType, FromUnit, ToUnit, Value):
    UnitType= Simplify(UnitType)
    try:
        Converter = Converters[UnitType]
    except:
        raise InvalidUnitTypeError(UnitType)
    return Converter.Convert(FromUnit, ToUnit, Value )
    
Convert = convert # so to have the old, non-PEP8 compatible name

### This is used by TapInput
def GetUnitAbbreviation(Units):
    # Abbreviation for units is a first element of synonyms list
    try:
        (UnitType,unit) = NameTable[Units]
    except KeyError:
        raise UnitConversionError("%s is not in UnitConverter Code"%Units)

    return(ConvertDataUnits[UnitType][unit][1][0])


## The exceptions
class UnitConversionError(Exception):
    """
    Exception type for unit conversion errors

    perhaps this should be subclassed more, but at the moment, I just pass a message back

    """
    def __init__(self,message):
        self.message = message

    def __str__(self):
        return self.message

class InvalidUnitError(UnitConversionError):
    """
    Exception raised when a unit is not in the Unit conversion database

    """
    def __init__(self, unit, type = ""):
        self.unit = unit
        self.type = type

    def __str__(self):
        return "The unit: %s is not in the list for Unit Type: %s"%(self.unit, self.type)

class InvalidUnitTypeError(UnitConversionError):
    """
    Exception raised when a unit is not in the Unitconversion database

    """
    def __init__(self, unitType):
        self.unitType = unitType

    def __str__(self):
        return "The unit type: %s is not in the UnitConversion database"%self.unitType
    
class MismatchedUnitError(UnitConversionError):
    """
    Exception raised when a unit is not in the Unitconversion database

    """
    def __init__(self, FromUnit, FromUnitType, ToUnit, ToUnitType):
        self.FromUnit     =  FromUnit    
        self.FromUnitType =  FromUnitType
        self.ToUnit       =  ToUnit      
        self.ToUnitType   =  ToUnitType  

    def __str__(self):
        return "The unit: %s of  type %s is not compatible with %s of type %s"% \
               (self.FromUnit, self.FromUnitType, self.ToUnit, self.ToUnitType)
