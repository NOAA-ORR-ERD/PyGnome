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

Other formats, such as GeoJSON and Shape files may be supported in the future.

The BNA format consists of a list of features: lines and polygons. GNOME uses polygons to define the land.

Each feature is preceded by a description line, such as the line shown below.

.. code-block:: none

   "2","1",21

The first number in quotes represents an identifier for the feature, and is usually unique. GNOME ignores the actual identifier in most cases.

The second number in quotes identifies the type of feature: ``"1"`` is a land feature;
``"2"`` is a water feature, or a polygon within another larger polygon (usually a lake).

The third number is the number of vertices of the feature, in order.
A positive number indicates a polygon.
Points are usually defined in a clockwise direction as you trace the land boundary (as though you are walking on an imaginary beach with your left foot on land and your right foot in the water).  However, PyGNOME ignore the orientation). The first point should be duplicated as the last point, so that a square would have 5 points defined.

A negative number of vertices defines a polyline, which GNOME will ignore.
Each vertex is defined on its own line as a longitude, latitude pair in decimal degrees, separated by a comma.

Example (course map of San Diego Bay): :download:`SanDiegoBay.bna <./SanDiegoBay.bna>`.

.. code-block:: none

  "2","1",21
  -117.133606, 32.580000
  -117.138306, 32.620889
  -117.180861, 32.681944
  -117.228361, 32.687250
  -117.223528, 32.708250
  -117.191778, 32.716611
  -117.163750, 32.698889
  -117.167444, 32.678833
  -117.154167, 32.680667
  -117.121778, 32.603722
  -117.098222, 32.615139
  -117.100889, 32.629417
  -117.120083, 32.673556
  -117.174139, 32.709750
  -117.178778, 32.727667
  -117.225806, 32.725028
  -117.247000, 32.667472
  -117.251897, 32.746000
  -117.049000, 32.746000
  -117.049000, 32.580000
  -117.133606, 32.580000


Two special types of polygons are defined for GNOME maps:

1. A map boundary: the out bounds of the model domain.

2. A spillable area: the region in which a spill can be initialized

If not supplied, the map boundary is the bounding box of the land polygons, and the spillable area is the entire map (excluding land).

These special polygons are most commonly used in Location Files to help users avoid setting spills in areas where the Location File has not been set up or well calibrated.

Map Bounds
..........

The map bounds define a polygon with a format similar to that shown above, but with the identifier: Map Bounds". This polygon should be the first or one of the last polygons in the map file.

.. code-block:: none

  "Map Bounds", "2", 6
  -117.200, 32.580
  -117.310, 32.700
  -117.310, 32.746
  -117.049, 32.746
  -117.049, 32.580
  -117.200, 32.580

Spillable Area
..............

The spillable area defines a polygon so that the user may not start spills outside the polygon, or over land areas within the polygon.
Again, the format is similar to other polygons in the bna format with the identifier: "Spillable Area".
This polygon should be one of the first or last defined in the map file.

.. code-block:: none

  "Spillable Area", "1", 6
  -117.150, 32.600
  -117.100, 32.600
  -117.100, 32.720
  -117.250, 32.720
  -117.250, 32.675
  -117.150, 32.600



