.. _units:

Units used in GNOME / PyGNOME
===========================

GNOME is a model of physical processes, and thus most of the things computed are in various physical units.

To keep things clean, the internal computational code is all handled in a consistent set of units.

conversion is done on I/O or occasionally within a class (i.e the class may store data in the units the user gave, but return values in the standard unit to other parts of the code) So unless otherwise noted, pass data into methods in the following standard units:

Time
    Time is expressed in integer seconds -- stored in a C ``unsigned long``
    
    Date-times are in seconds since 1904 (1904-01-01T00:00) -- stored in a C ``unsigned long``
   
    (NOTE: much of the Python code uses ``datetime.datetime`` objects and/or numpy ``datetime64`` objects, but the internal C++ code used integer seconds)

Length
    Lengths are in floating point meters   

Mass
    Mass is in floating point grams ???

Velocity
    Velocities are in meters per second (usually ``double``)
    
Latitude-Longitude
   Lat-long is in floating point (``double``) degrees  -- range generally -360 to 360, so we can do stuff accross the date line.

   





