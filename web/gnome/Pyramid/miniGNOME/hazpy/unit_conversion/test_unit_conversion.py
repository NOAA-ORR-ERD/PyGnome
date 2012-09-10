#!/usr/bin/env python

import unit_conversion
import unittest

KnownValues = [
    ## Known values from Handbook of Chemistry and Physics, except where noted
    ("length", "meters", "feet", 1, 3.2808398),
    ("length", "feet", "meters", 1, .3048),
    ("length", "feet", "miles", 1, 0.000189393),
    ("length", "feet", "nauticalmiles", 1, .00016457883),
    ("length", "feet", "inches", 1, 12),
    ("length", "fathom", "cm", 1, 182.88),
    ("length", "Latitude Minutes", "NauticalMiles", 1.0, 1.0 ),
    ("Length", "LatitudeDegrees", "NauticalMiles", 1.0, 60 ),
    ("Length", "micron", "mm", 100, .1 ),
    ("Length", "km", "yard", 1, 1093.6133 ),
   

    # all values close to value in the "Open Water Oil Identification Job Aid"
    # and close to values in the Unit Conversion sheet distributed with the dispersion mission planner.
    ("Oil Concentration", "micron", "mm", 100, .1 ),
    ("Oil Concentration", "in", "mm", 1.0, 25.4 ),
    ("Oil Concentration", "micron", "bbl/acre", 1.0, 0.02545396 ), # calculated from HCP -- 
    ("Oil Concentration", "bbl/acre", "m^3/km^2", 1.0, 39.2866 ), # calculated from HCP -- 
    ("Oil Concentration", "bbl/acre", "bbl/sq.mile", 1.0, 640.0 ), # calculated from HCP -- 
    ("Oil Concentration", "gal/acre", "bbl/acre", 42.0, 1.0 ), # calculated from HCP -- 
    ("Oil Concentration", "m^3/km^2", "liter/hectare", 1, 10.0 ), # calculated from HCP -- 

    ("Area", "sq m","ft^2", 10, 107.63910),
    ("Area", "Acre","square yards", 1, 4840),
    ("Area", "Hectares","cm^2", 1.0, 1e8),
    ("Area", "ha","sq miles", 1.0, 0.0038610216),
    ("Area", "ft^2","in^2", 1.0, 144),
    ("Area", "km^2","ft^2", 1.0, 1.0763910e7),
    ("Area", "nm^2", "acre", 1.0, 847.547741), # calculated from HCP values
    ("Area", "nm^2", "square mile", 1.0, 1.3242932), # calculated from HCP values


    ("Volume", "liters","gal", 1.0, 0.26417205),
    ("Volume", "cubicmeters","gal", 0.0037854118, 1.0),
    ("Volume", "milliongallons","gal", 1.0, 1e6),
    ("Volume", "liters","ft^3", 1.0, 0.035314667),
    ("volume", "bbl","l", 1.0, 158.9873),
    ("volume", "cubicinches", "cubicfeet", 1.0, 0.00057870370),
    ("volume", "cc", "cubicyard", 1.0, 1.3079506e-6),
    ("volume", "fluid ounce (UK)", "oz", 1.0, 0.9607594),
    ("volume", "gallon (UK)", "gal", 1.0, 1.200949),

    ("mass", "kg","lb", 1.0, 2.2046226),
    ("Mass", "kg","metrictons", 1.0, 0.001),
    ("mass", "kg","lb", 1.0, 2.2046226),
    ("mass", "pound","slug", 1.0, 0.0310810),
    ("mass", "ounce","lbs", 1.0, 0.0625),
    ("mass", "ton","gram", 1.0, 907184.74),
    ("mass", "ton(UK)","ton", 1.0, 1.12),

    ("Time", "seconds","minutes", 60, 1.0),
    ("time", "days","minutes", 1.0, 24*60),
    ("time", "hr","seconds", 1.0, 60*60),

    ("Velocity", "m/s","cm/s", 1.0, 100),
    ("Velocity", "km/h","kts", 1.0, 0.5399568),
    ("Velocity", "mph","ft/s", 1.0, 1.4666666),
    ("Velocity", "ft/min","mph", 1.0, 0.01136363),
    ("Velocity", "ft/s","m/min", 1.0, 18.288),
    ("Velocity", "ft/hr","cm/s", 1.0, 0.0084666),
    ("Velocity", "ft/hr","km/hr", 1.0, 0.0003048),
    ("Velocity", "knot","m/s", 1.0, 0.514444),

    ("Discharge", "cfs","l/s", 1.0, 28.31685),
    ("Discharge", "gal/hr","gal/min", 60.0, 1.0),
    ("Discharge", "bbl/day","l/s", 1.0, (158.9873/24/3600)), # calculated from bll=>liter
    ("Discharge", "ft^3/min","gal/min", 1.0, 7.4805195), 
    ("Discharge", "m^3/min", "m^3/s", 1.0, 1.0/60.0),  
    ("Discharge", "m^3/min", "gal/min", 1.0, 264.1721),  
    ("Discharge", "gal/hr","m^3/min", 1.0, 6.3090197e-5),  
    ("Discharge", "gal/hr","m^3/hr", 1.0, 6.3090197e-5*60.0), # calculated from m^3/min  
    ("Discharge", "bbl/day","gal/day", 1.0, 42), # from definition of bbl  
    ("Discharge", "l/min","cfs", 1.0, 0.000588578), 
    ("Discharge", "bbl/hr","cfs", 1.0, 5.614583/3600.), # calculated from cfs 
    ("Discharge", "cfs","gal/sec", 1.0, 448.83117/60), # calculated from gal/min 


    ("Kinematic Viscosity", "stokes", "cSt", 1, 100.0),
    ("Kinematic Viscosity", "cm^2/s", "m^2/s", 1, .0001),
    ("Kinematic Viscosity", "cm^2/s", "square inch per second", 1, 0.15500031),
    ("Kinematic Viscosity", "SSU", "cSt", 462.0, 100.0),
    ("Kinematic Viscosity", "SSF", "cSt", 342.0, 724.0),
    ("Kinematic Viscosity", "mm^2/s", "cSt", 1.0, 1.0), # from ASTM report

    ("temperature", "F","C", 32, 0),
    ("temperature", "F","C", 212, 100),
    ("temperature", "C","K", 0, 273.16),
    ("temperature", "F","K", 32, 273.16),
    
    ("density", "g/cm^3","Specific Gravity (15C)", 1, (1/0.99913)),
    ("density", "g/cm^3","Specific Gravity (15C)", 0.99913, 1),
    ("density", "Specific gravity","gramspercubiccentimeter", 1, 0.99913),
    ("density", "SG","API", 1.0, 10),
    ("density", "SG","API", 2.0, -60.75),
    ("density", "SG","API", 0.9, 25.7222),
    ("density", "API","SG", 25.7222, 0.9),
    ("density", "lbs/ft^3","Grams per Cubic Centimeter", 1.0, 0.016018463),

    ("Concentration In Water", "ppb","ppm", 1000, 1), # calculated
    ("Concentration In Water", "fraction","%", 1, 100), # calculated
    ("ConcentrationInWater", "kg/m^3","lb/ft^3", 16.018464, 1), # calculated
    ("concentrationinwater", "mg/l","ppb", 1.0, 1000), # calculated
    ("ConcentrationInWater", "ppt","percent", 1.0, .1), # calculated
    ("ConcentrationInWater", "ug/l", "ppb", 1.0, 1.0), # calculated
    ("ConcentrationInWater", "nanogramperliter", "partpertrillion", 1.0, 1.0), # calculated
    ]

def Close(a, b, Epsilon = 1e-5):
    # Is this accurate enough? The input data doesn't have a lot of sig figs.
    """
    Close(a,b) returns true is a and b are the same within Epsilon

    """
    if a == b:
        return True
    else:
        return abs( float(a - b) / ((a+b)/2) ) <= Epsilon

# nose generater for known values tests
def test_known_values():
    for val in KnownValues:
        yield check_known_value, val

def check_known_value(test):
#        print "testing:", test
        Type = test[0]
        From = test[1]
        To = test[2]
        Value = test[3]
        True = test[4]
        Calculated = unit_conversion.Convert(*test[:4])
        assert(Close(Calculated, True))
#               ("Test of %s:%s to %s failed. From: %g, To: %.10g, Got %.10g"%
#                (Type, From, To, Value, True, Calculated ) )
#               )

#def CreateKnownValueTest(KnownValue):
#    def test(self, test=KnownValue):
#        print "testing:", test
#        Type = test[0]
#        From = test[1]
#        To = test[2]
#        Value = test[3]
#        True = test[4]
#        Calculated = unit_conversion.Convert(*test[:4])
#        self.failUnless(Close(Calculated, True),
#                        ("Test of %s:%s to %s failed. From: %g, To: %.10g, Got %.10g"%
#                         (Type, From, To, Value, True, Calculated ) )
#                        )
        
#class testKnownValues(unittest.TestCase):
#    pass    
#    def testKnown(self):
#        for test in KnownValues:
#            Type = test[0]
#            From = test[1]
#            To = test[2]
#            Value = test[3]
#            True = test[4]
#            Calculated = unit_conversion.Convert(*test[:4])
#            self.failUnless(Close(Calculated, True),
#                            ("Test of %s:%s to %s failed. From: %g, To: %.10g, Got %.10g"%
#                             (Type, From, To, Value, True, Calculated ) )
#                            )

## add tests to testKnownValues class
#for i, KnowValue in enumerate(KnownValues):
#    name = "testKnown%i"%i
#    setattr(testKnownValues, name, CreateKnownValueTest(KnowValue))

#print dir(testKnownValues)

#T = testKnownValues()
#T.testKnown27()

class testBadnames(unittest.TestCase):
    def testBadType(self):
        self.failUnlessRaises(unit_conversion.InvalidUnitTypeError,
                          unit_conversion.Convert,
                          "BadType","feet","miles", 0,
                          )
    def testBadUnit1(self):
        self.failUnlessRaises(unit_conversion.InvalidUnitError,
                          unit_conversion.Convert,
                          "Length","eggs","miles", 0,
                          )
    def testBadUnit2(self):
        self.failUnlessRaises(unit_conversion.InvalidUnitError,
                          unit_conversion.Convert,
                          "Length","feet","spam", 0,
                          )
    def testBadUnit3(self):
        self.failUnlessRaises(unit_conversion.InvalidUnitError,
                          unit_conversion.Convert,
                          "Density","API","feet", 0,
                          )
class testOilQuantityConverterClass(unittest.TestCase):
    OQC = unit_conversion.OilQuantityConverter
    
    def testMassToVolume1(self):
        self.failUnless(Close(self.OQC.ToVolume(Mass=1,
                                                MassUnits="metricton",
                                                Density=25,
                                                DensityUnits="API",
                                                VolumeUnits="bbl"),
                              6.9626324)
                        )
                         
    def testMassToVolume2(self):
        self.failUnless(Close(self.OQC.ToVolume(Mass=1,
                                           MassUnits="metricton",
                                           Density=0.816,
                                           DensityUnits="SG",
                                           VolumeUnits="bbl"),
                         7.71481307)
                        )
    def testVolumeToMass1(self):
        Expected = 1.0
        Calculated = self.OQC.ToMass(Volume=6.9626324,
                                     VolUnits="bbls",
                                     Density=25,
                                     DensityUnits="API",
                                     MassUnits="metricton")
        print Expected, Calculated
        self.failUnless(Close(Expected, Calculated))

    def testVolumeToMass1(self):
        Expected = 1.0
        Calculated = self.OQC.ToMass(Volume=7.83861191,
                                     VolUnits="bbls",
                                     Density=0.816,
                                     DensityUnits="SG",
                                     MassUnits="longton")
        #print Expected, Calculated
        self.failUnless(Close(Expected, Calculated))

class testGetUnitType(unittest.TestCase):
    pass

if __name__ == "__main__":
    unittest.main()

