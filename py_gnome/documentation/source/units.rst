.. _units:

#############################
Units used in GNOME / PyGNOME
#############################

GNOME is a model of physical processes, and thus most of the things computed are in various physical units.

To keep things clean, the internal computational code is all handled in a consistent set of units. All values are stored in floating point types, generally 64 bit (``double``) unless otherwise noted.

Conversion is done on I/O or occasionally within a class (i.e the class may store data in the units the user gave, but return values in the standard unit to other parts of the code) So unless otherwise noted, pass data into methods in the following standard units:

Time
    In most places, time is represented by Python `datetime` and `timedelta` objects.

    In a few cases (model timestep for instance) time is expressed in integer seconds

    IN the C / C++ code, time is stored as seconds in a C ``unsigned long``, and datetimes are in seconds since 1904 (1904-01-01T00:00) -- stored in a C ``unsigned long``

Length
    Lengths are in meters

Mass
    Mass is in  grams

Volume
    Volume is in cubic meters

Density
    Density is in grams per cubic centimeter (g/cm^3)

Velocity
    Velocities are in meters per second

Latitude-Longitude
   Lat-long is in floating point degrees  -- range generally -360 to 360, so we can do stuff across the date line.

Diffusion Coefficients
   Diffusion coefficients (for both vertical and horizontal random diffusion) are given in units of square centimeters per second cm^2/s

Droplet Diameter
   Droplet Diameter is given and returned in meters

Viscosity
   Viscosity is usually kinematic viscosity, and is in units of m^2/s

Salinity
   Salinity is used in various calculations for sedimentiaon, wave formation, etc. Standard units is Practical Salinity Units (PSU) -- more or less parts per thousand -- i.e. fresh is 0, typical seawater is 35.





