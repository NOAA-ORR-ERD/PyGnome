.. _units:

Units used in GNOME / PyGNOME
===========================

GNOME is a model of physical proceses, and thus most of the thigns computed are in various physical units.

to keep things clean, the internal computational code is all handled in a consitent set of units.

conversion is done on I/O or ocastionally within in a class (i.e the class may store data in the units the user gave, but return values in the standard unit to other parts of the code) So unless otherwise noted, pass data into methods in the following standard units:

Time
    Time is expressed in floating point seconds -- stored in a C ``unsigned long``
    
    Date-times are in seconds since 1904 (1904-01-01T00:00) -- stored in a C ``unsigned long``
   
Length
    Lengths are in floating point meters   

Mass
    Mass is in floating point grams ???

Velocity
    Velocities are in meters per second (usaully ``double``)
    
Latitude-Longitude
   Lat-long is is floating point (``double``) degrees  -- range generally -360 to 360, so we can do stuff accross the date line.

   





