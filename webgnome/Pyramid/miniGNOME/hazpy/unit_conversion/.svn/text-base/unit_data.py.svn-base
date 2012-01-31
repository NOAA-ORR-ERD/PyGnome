#!/usr/bin/env python

"""
unit_data.py

This is the data used by unit_conversion

it also includes a utility function for dumping the units to the console or a file:
   write_units(filename=None)

"""
ConvertDataUnits = {
# All lengths in terms of meter
# All conversion factors from "Handbook of Chemistry and Physics" (HCP) except where noted.
    
"Length" : {"meter"      : (1.0,["m","meters","metre"]),
            "centimeter" : (0.01,["cm", "centimeters"]),
            "millimeter"  : (0.001,["mm","millimeters"]),
            "micron"  : (0.000001,["microns"]),
            "kilometer"  : (1000.0,["km","kilometers"]),
            "foot"        : (0.3048,["ft", "feet"]),
            "inch"      : (0.0254,["in","inches"]),
            "yard"       : (0.9144,[ "yrd","yards"]),
            "mile"       : (1609.344,["mi", "miles"]),
            "nautical mile" : (1852.0,["nm","nauticalmiles"]),
            "fathom"  : (1.8288,["fthm", "fathoms"]),
            "latitude degree": (111120.0,["latitudedegrees"]),
            "latitude minute": (1852.0,["latitudeminutes"])
            },

# this is technically length but used differently, so I'm keeping it separate
# micron is the base unit
"Oil Concentration" : {"micron"         : (1.0, ["microns"]),
                       "cubic meter per square kilometer" : (1.0,["m^3/km^2",]),
                       "millimeter"     : (1000.,["mm","millimeters"]),
                       "inch"           : (25400.,["in","inches"]),
                       "barrel per acre": (39.2866176,["bbl/acre",]), # calculated from HCP
                       "barrel per square mile": (0.06138533995, ["bbl/sq.mile",]), # calculated from HCP
                       "gallon per acre": (0.93539563202687404,["gal/acre",]), # calculated from HCP
                       "liter per hectare": (0.1,["liter/hectare",]), # calculated from HCP
                       },

# All Areas in terms of square meter
"Area" : {"square meter"  : (1.0,["m^2","sq m","squaremeter"]),
          "square centimeter": (.0001,["cm^2","sq cm"]),
          "square kilometer"  : (1e6,["km^2","sq km","squarekilometer"]),
          "acre"  : (4046.8564,["acres"]),
          "square mile"  : (2589988.1,["sq miles","squaremile"]),
          "square nautical mile"  : (3429904,["sq nm","nm^2"]), # calculated from HCP
          "square yard"  : (0.83612736,["sq yards","squareyards"]),
          "square foot"  : (0.09290304,["ft^2", "sq foot","square feet"]),
          "square inch"  : (0.00064516,["in^2", "sq inch","square inches"]),
          "hectare"  : (10000.0,["hectares","ha"]),
          },

# All volumes in terms of cubic meter
"Volume" : {"cubic meter"  : (1.0,["m^3","cu m","cubic meters"]),
            "cubic centimeter"  : (1e-6,["cm^3","cu cm", "cc"]),
            "barrel (petroleum)" : (.1589873,["bbl","barrels","barrel","bbls",]),
            "liter"        : (1e-3,["l","liters"]),
            "gallon"       : (0.0037854118, ["gal","gallons","gallon","usgal"]),
            "gallon (UK)"  : (0.004546090, ["ukgal","gallons(uk)"]),
            "million US gallon"  : (3785.4118, ["milliongallons","milgal"]),
            "cubic foot"    : (0.028316847, ["ft^3","cu feet","cubicfeet"]),
            "cubic inch"    : (16.387064e-6, ["in^3","cu inch","cubicinches"]),
            "cubic yard"    : (.76455486, ["yd^3","cu yard","cubicyard","cubicyards"]),
            "fluid ounce"      : (2.9573530e-5, ["oz","ounces(fluid)", "fluid oz"]),
            "fluid ounce (UK)" : (2.841306e-5, ["ukoz", "fluid oz(uk)"]),
            },

# All Temperature units in K (multiply by, add)
"Temperature" : {"Kelvin"  : ((1.0, 0.0),["K","degrees k","degrees k","degrees kelvin","degree kelvin","deg k"]),
                 "Celsius"  : ((1.0, 273.15),["C","degrees c","degrees celsius","deg c","centigrade"]),
                 "Farenheight"  : ((0.55555555555555558, (273.16*9/5 - 32) ),["F","degrees f","degree f","degrees farenheight","deg f"]),
                 },

# All Mass units in Kg (weight is taken to be mass at standard g)
"Mass" : {"kilogram"  : (1.0,["kg","kilograms"]),
          "pound"     : (0.45359237,["lb","pounds","lbs"]),
          "gram"  : (.001,["g","grams"]),
          "ton"   : (907.18474, ["tons","uston"]),
          "metric ton (tonne)" : (1000.0, ["tonnes","metric ton","metric tons"]),
          "slug"       : (14.5939, ["slugs"]),
          "ounce"       : (.028349523, ["oz","ounces"]),
          "ton (UK)"       : (1016.0469, ["ukton","long ton"]),
          },

# All Time In second
"Time" : {"second"  : (1.0,["sec","seconds"]),
          "minute"  : (60.0,["min","minutes"]),
          "hour"    : (3600.0,["hr","hours","hrs"]),
          "day"     : (86400.0,["day","days"]),
          },
# All Velocities in meter per second
"Velocity" : {"meter per second"  : (1.0,["m/s","meters per second","mps"]),
              "meter per minute" : (0.01666666666, ["m/min", "meters per minute"]),
              "centimeter per second"  : (.01,["cm/s"]),
              "kilometer per hour"  : (0.277777,["km/h", "km/hr"]),
              "knot"  : (0.514444,["knots","kts"]),
              "mile per hour"  : (0.44704,["mph","miles per hour"]),
              "foot per second"  : (0.3048,["ft/s", "ft/sec", "feet per second", "feet/s"]),
              "foot per minute"  : (0.00508,["ft/min", "feet per minute", "feet/min"]),
              "foot per hour"  : (0.000084666, ["ft/hr", "feet per hour", "feet/hour"]),
              },
# All Discharges in cubic meter per second
"Discharge" : {"cubic meter per second"  : (1.0, ["m^3/s","cu m/s","cms"]),
               "cubic meter per min"  : (1.0/60., ["m^3/min",]),
               "cubic meter per hour"  : (1.0/3600.0, ["m^3/hr",]),
               "liter per second"    : (0.001, ["l/s","lps"]),
               "liter per minute"    : (0.001/60, ["l/min",]),
               "cubic foot per second"  : (.02831685, ["cfs","cu feet/s","feet^3/s"]),
               "cubic foot per minute"  : (0.00047194744, ["ft^3/min"]),# calculated from cm^3/s
               "gallon per day"  : (4.3812636805555563e-08, ["gal/day"]),# calculated from gal/hr
               "gallon per hour"  : (1.0515032833333335e-06, ["gal/hr"]),
               "gallon per minute" : (6.3090197000000006e-05, ["gal/min", "gpm"]),
               "gallon per second"  : ( 0.0037854118, ["gal/s", "gal/sec"]),
               "barrel per hour"  : ( 4.4163138888888885e-05, ["bbl/hr"]),
               "barrel per day"  : ( 1.84013078e-06, ["bbl/day"]), # calculated from bbl/hr
               },

### Kinematic Viscosity in Stokes
##  NOTE: there is a more detailed way to do this, specified in:
##        ASTM D 2161 Standard Practice for Conversion of Kinematic
##        Viscosity to Saybolt Universal Viscosity or to Saybolt Furol
##        Viscosity
## for the moment, this will only handle approximation for SFS and SUS
"Kinematic Viscosity" : {"Stoke": (1.0, ["St","stokes"]),
                         "centiStoke": (.01, ["cSt","centistokes"]),
                         "square millimeter per second": (.01, ["mm^2/s",]),
                         "square centimeter per second": (1.0, ["cm^2/s",]),
                         "square meter per second": (10000, ["m^2/s"],),
                         "square inch per second": (6.4516, ["in^2/s","squareinchespersecond"]),
                         "Saybolt Universal Second": (1/462.0, ["SSU","SUS"]),# from CRC - only good for > 100cSt
                         "Saybolt Furol Second": (0.02116959064, ["SSF","SFS"]),# from Fuel Oil Manual: good for 724cSt
                         #"poise" : (["P"])
                         },

### Density in g/cc
## NOTE: Specific Gravity can only be defined for a given reference temperature.
##       The most common standard in the oil industry is 15C (or 60F). The
##       following is the value for the Density of water at 15C
##       (CRC Handbook of Chemistry and Physics) 
"Density" : {"gram per cubic centimeter"  : (1.0,["g/cm^3","grams per cubic centimeter"]),
             u"specific gravity (15\xb0C)"  : (0.99913,["S","specificgravity","Spec grav","SG","specificgravity(15C)"]),
             "kilogram per cubic meter" : (.001,["kg/m^3"]),
             "pound per cubic foot":  (0.016018463,["lbs/ft^3"]),
             "API degree"  : (1,["api"]),# this is special cased in the code.
             },

### Concentration in water in PPM
"Concentration In Water" : {"part per million"  : (1.0,["ppm","parts per million"]),
                            "part per billion"  : (.001,["ppb", "parts per billion"]),
                            "part per thousand" : (1000,["ppt", "parts per thousand"]),
                            "part per trillion" : (.000001,["parts per trillion","pptr"]),
                            "fraction (decimal)" : (1e6,["fraction", "mass per mass"]),
                            "percent"  : (1e4,["%", "parts per hundred", "per cent"]),
                            "kilogram per cubic meter":  (1000,["kg/m^3","kg/m3"]),
                            "pound per cubic foot": (16018.463, ["lb/ft^3"]),
                            "milligram per liter": (1.0, ["mg/l"]),
                            "microgram per liter": (0.001, ["ug/l"]),
                            "nanogram per liter": (0.000001, []),
                          }
}

def write_units(filename=None):
    import sys
    if filename:
        f = open(filename, 'w')
    else:
        f = sys.stdout
    f.write("NUCOS unit set:\n")
    for key, value in ConvertDataUnits.items():
        f.write( "\n%s:\n"%key )
        for key in value:
            f.write( "    %s\n"%key.encode('ascii', 'ignore') )
             


