.. _map_formats:

Maps
====

In order to model the interaction of particles with land, GNOME requires a definition of where the land is.
This definition is provided as polygons surrounding land, in lat-lon coordinates (usually WGS-84, but as long as the coords are the same as for other input data, it will work).
Longitude can be either in 0--360 degrees, or in -180--180 degrees, as long as the currents and winds used are in the same coordinate system.

BNA format
----------

Currrently, the only format supported by GNOME is the
`BNA format <https://www.softwright.com/faq/support/boundary_file_bna_format.html>`_ BNA is a fairly simple ASCII text format, that can be hand-created or converted from other formats.

Other formats, such as geoJSON and Shapefiles may be supported in the future.

BNA format consists of a list of lines and polygons that are to be drawn on the screen. Each feature is preceded by a description line, such as the line shown below, from the example file ``simple.bna``.

::
  "2","1",18

The first number in quotes represents an identifier for the feature, and is usually unique.
The second number in quotes identifies the type of feature: ``"1"`` is a land feature;
``"2"`` is a water feature, or a polygon within another larger polygon ( usually a lake).
The third number is the number of points in the feature, in order for drawing.
A positive number indicates a polygon.
Points are defined in a clockwise direction as you trace the land boundary (as though you are walking on an imaginary beach with your left foot on land and your right foot in the water). (Note that PyGNOME ignore the orientation).
A negative number defines a line where the start and end points don’t connect.

::
   File Name: simple.bna
   "2","1",18
   -82.521416,27.278500
   -82.552109,27.353674
   -82.564636,27.383394
   -82.600746,27.500633
   -82.576721,27.581442
   -82.541473,27.665442
   -82.478104,27.725504
   -82.443367,27.755222
   -82.250000,27.730673
   -82.250000,27.685675
   -82.250000,27.640678
   -82.250000,27.595680
   -82.250000,27.505688
   -82.250000,27.460690
   -82.250000,27.415693
   -82.250000,27.370695
   -82.351616,27.278500
   -82.453232,27.278500
   "2","1",10
   -82.250000,27.865969
   -82.333580,27.864744
   -82.383003,27.879385
   -82.479012,27.888107
   -82.543144,27.952902
   -82.456032,28.066999
   -82.405220,28.066999
   -82.354408,28.066999
   -82.250000,27.977007
   -82.250000,27.898989

Two special types of polygons are defined for GNOME maps: (1) a map boundary for nonrectangular maps and (2) a spillable area. These special polygons are most commonly used in Location Files to help users avoid setting spills in areas where the Location File has not been set up or well calibrated.

Map Bounds
..........

The map bounds define a polygon with a format similar to that shown above. This polygon should be the first polygon in the map file.

"Map Bounds","1",7 -121.319176,35.199476 -121.319176,34.197944 -121.218496,34.0 -119.378944,34.0 -119.221448,34.152428 -119.221448,35.199476 -121.319176,35.199476

Spillable Area
..............

The spillable area defines a polygon so that the user may not start spills outside the polygon, or over land areas within the polygon. Again, the format is similar to other polygons in the bna format. This polygon should be the last one defined in the map file.

"SpillableArea", "1", 15 -121.319176,35.199476 -121.319176,34.197944 -121.218496,34.0 -120.633640,34.0 -120.445584,34.088112 -120.381776,34.085196 -120.204512,34.026884 -120.066248,34.053124 -119.931528,34.061872 -119.729456,34.015220 -119.534464,34.047292 -119.378944,34.0 -119.221448,34.152428 -119.221448,35.199476 -121.319176,35.199476



