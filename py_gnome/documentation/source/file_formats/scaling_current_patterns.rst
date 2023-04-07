.. _scaling_current_patterns:

Scaling Current Patterns
========================

Since the current patterns created in CATS only indicate the direction of the current and the relative speeds, these current patterns need to be scaled in order to be useful with the trajectory model.
For example, consider a fictitious current pattern with only two triangles, A and B. The velocity in triangle A is 1.2 to the east and the velocity in triangle B is 1.8 to the north.
Observations indicate that the velocity in triangle A should be 3.0 knots to the east, so we must scale the current pattern by the ratio of these velocities in triangle A, or (3.0 knots ÷ 1.2 = 2.5 knots).
That is, multiplying the velocity in triangle A in the current pattern (1.2) by the scale factor (2.5 knots) yields the observed velocity (3.0 knots).
The direction did not change.
To find the velocity in triangle B, we multiply the velocity in triangle B in the current pattern (1.8) by the scale factor (2.5 knots) to get a velocity of 4.5 knots. The velocity in triangle B is still to the north, since the direction does not change in the current pattern.
GNOME is quite helpful in scaling current patterns. At a given reference point in the current pattern, GNOME tells you what the flow is.
You then input into GNOME what you would like the velocity to be at the reference point, and GNOME calculates the scaling coefficient for the pattern for you!

The direction of the flow in the current fields in GNOME can reverse by multiplying the pattern by a negative scaling coefficient. The ebb and flow of tides are simulated this way, through a time-series of positive and negative scaling values. You can scale currents with either a constant value or a time-series. The acceptable file formats for time-series are outlined below.

Time-Series File Formats
------------------------

Current patterns in GNOME can be scaled to be time dependent with two different file types:

1. a time-series of current magnitude or
2. a “SHIO mover” that contains data for GNOME to use in calculating tidal current magnitudes.

All data in this section were created by the NOAA SHIO application (“shio” comes from Japanese for “tide”).

The South Bend, Washington, U.S.A. station on the Willapa River was chosen for all the examples in this section. Below is the information found in the `SouthBend.text` file to illustrate the information GNOME needs in order to calculate the tidal currents at this station. This particular file is not a data file for GNOME. Those data are represented in data files presented later in this discussion.

Example – Filename: SouthBend.text
..................................

.. code-block:: none

    Tidal currents at South Bend, Willapa River, WASHINGTON COAST
    Station No. CP1009
    Meter Depth: n/a

    Latitude: 46˚0' N
    Longitude: 123˚47' W

    Maximum Flood Direction: 90˚
    Maximum Ebb Direction: 270˚

    Time offsets    Hour:Min
    Min Before Flood    0:19am
    Flood   0:20am
    Min Before Ebb  0:24am
    Ebb  -0:06am

    Flood Speed Ratio: 0.6
    Ebb Speed Ratio: 0.5

        Speed(kt)   Direction(deg.)
    Min Before Flood    00.0    n/a
    Flood   01.2    090
    Min Before Ebb  00.0     n/a
    Ebb 01.4    270

    Based on Grays Harbor Ent.

    Local Standard Time

    ------------------------------------------------------------------
    Mon, Aug 24, 1998--Sunrise -- 6:09am  Sunset -- 7:55pm
    0:38am   +01.2  Max Flood
    3:31am  +00.0   Min Before Ebb
    6:29am  -01.6   Max Ebb
    9:59am  +00.0   Min Before Flood
    1:08pm  +01.4   Max Flood
    4:12pm  +00.0   Min Before Ebb
    6:56pm  -01.4   Max Ebb
    10:17pm +00.0   Min Before Flood



Time Series of Current Magnitude
................................

Time series file for currents have the format

``dd, mm, yy, hr, min, u, 0.0``

where dd is the day, mm is the month, yy is the year, hr is the hour, min is the minute, u is the magnitude of the velocity, and 0.0 is a number to indicate that the file is in a magnitude format rather than a U,V format. The direction is left blank because the current pattern supplies the individual current vectors. There is an optional header:

- The first line lists the station name.
- The second line lists the station position.
- The third line provides the units.

For example, the ``SouthBend.ossm`` file contains one day of tidal information for South Bend, Washington, U.S.A.

Example – Filename: SouthBend.ossm
..................................


.. code-block:: none

    South Bend
    -123.78,46
    knots
    24, 8, 98, 0, 37, 1.2, 0.0
    24, 8, 98, 3, 30, 0.0, 0.0
    24, 8, 98, 6, 28, -1.6, 0.0
    24, 8, 98, 9, 58, 0.0, 0.0
    24, 8, 98, 13, 7, 1.4, 0.0
    24, 8, 98, 16, 11, 0.0, 0.0
    24, 8, 98, 18, 55, -1.4, 0.0
    24, 8, 98, 22, 16, 0.0, 0.0

Annotated Version of the File

.. code-block:: none

    # Day,  Month,  Year,  Hour,  Min.,  Speed,  Direction # (Dummy Value)
    24,     8,    98,     0,     37,     1.2,  0.0
    24,     8,    98,     3,     30,     0.0,  0.0
    24,     8,    98,     6,     28,    -1.6,  0.0
    24,     8,    98,     9,     58,     0.0,  0.0
    24,     8,    98,     13,     7,     1.4,  0.0
    24,     8,    98,     16,    11,     0.0,  0.0
    24,     8,    98,     18,    55,    -1.4,  0.0
    24,     8,    98,     22,    16,     0.0,  0.0


SHIO Movers: Using Tidal Constituents
-------------------------------------

GNOME can use both tidal height and tidal current constituent data to scale current patterns. In the case of tidal height station data. GNOME will take the time derivative of the tidal heights, and request the user to scale that derivative to calculate the tidal currents. For the tidal current station data (below), GNOME will use the calculated currents directly. The constituent record data are rather complex, so we have provided information about the data fields and then provided an example file.

Tidal Heights Constituent Record:

.. code-block:: none

    Line 1  [StationInfo]
    Line 2  Type=H  station type for heights is “H”
    Line 3  staName station name
    Line 4  Latitude=latStr decimal degrees
    Line 5  Longitude=longStr   decimal degrees
    Line 6  [Constituents]
    Line 7  DatumControls.datum=datum   mean sea level
    Line 8  DatumControls.FDir=0    bug. Type as seen. Will be fixed in 1.2.7
    Line 9  DatumControls.EDir=0    bug. Type as seen. Will be fixed in 1.2.7
    Line 10 DatumControls.L2Flag=0  bug. Type as seen. Will be fixed in 1.2.7
    Line 11 DatumControls.HFlag=0   bug. Type as seen. Will be fixed in 1.2.7
    Line 12 DatumControls.RotFlag=0 bug. Type as seen. Will be fixed in 1.2.7
    Lines 13-17 constituent amplitudes in order     M2, S2, N2, K1, M4, O1, M6, MK3, S4, MN4, NU2, S6, MU2, 2N2, OO1, LAM2, S1, M1, J1, MM, SSA, SA, MSF, MF, RHO, Q1, T2, R2, 2Q1, P1, 2SM2, M3, L2, 2MK3, K2, M8, MS4 [feet]
    Lines 18-23 constituent phases in order see above, lines 13-17 [degrees]
    Line 24 [Offset]
    Note:  Lines 25-30 use a second integer to indicate to GNOME whether there is valid data in the field.  “0” indicates no data, so GNOME can skip the calculation.  “1” indicates valid data exists (that may be zero).
    Line 25 HighTime=highTime 1 high water time adjustment (minutes)
    Line 26 LowTime=lowTime 1   low water time adjustment
    Line 27 HighHeight_Mult=highHeightScalar 1  high water height multiplier
    Line 28 HighHeight_Add=highHeightAdd 1  high water height addend
    Line 29 LowHeight_Mult=lowHeightScalar 1    low water height multiplier
    Line 30 LowHeight_Add=lowHeightAdd 1    low water height addend


Example – Filename:  HornIslandPass.shio.txt
............................................

.. code-block:: none

    [StationInfo]
    Type=H
    Name=Horn Island Pass
    Latitude=30.2167
    Longitude=-88.483333
    [Constituents]
    DatumControls.datum=0.620000
    DatumControls.FDir=0
    DatumControls.EDir=0
    DatumControls.L2Flag=0
    DatumControls.HFlag=0
    DatumControls.RotFlag=0
    H=0.066000 0.022000 0.013000 0.468000 0.000000 0.460000 0.000000 0.000000 0.000000 0.000000 0.000000 0.000000 0.000000 0.000000 0.020000 0.000000 0.000000 0.033000 0.036000 0.000000 0.120000 0.299000 0.000000 0.000000 0.018000 0.099000 0.000000 0.000000 0.012000 0.139000 0.000000 0.000000 0.000000 0.000000 0.000000 0.000000 0.000000 0.000000 0.000000 0.000000 0.000000 0.000000
    kPrime=358.500000 355.700012 0.000000 327.000000 0.000000 324.200012 0.000000 0.000000 0.000000 0.000000 0.000000 0.000000 0.000000 0.000000 329.799988 0.000000 0.000000 325.500000 328.399994 0.000000 32.099998 151.800003 0.000000 0.000000 323.000000 314.100006 0.000000 0.000000 321.299988 331.899994 0.000000 0.000000 0.000000 0.000000 0.000000 0.000000 0.000000 0.000000 0.000000 0.000000 0.000000 0.000000
    [Offset]
    HighTime=-0.516667 1
    LowTime=-0.883333 1
    HighHeight_Mult=1.300000 1
    HighHeight_Add=0.000000 1
    LowHeight_Mult=1.300000 1
    LowHeight_Add=0.000000 1

Tidal Currents Constituent Record
.................................

.. code-block:: none

    Line 1  [StationInfo]
    Line 2  Type=C  station type for currents is “C”
    Line 3  staName station name
    Line 4  Latitude=latStr decimal degrees
    Line 5  Longitude=longStr   decimal degrees
    Line 6  [Constituents]
    Line 7  DatumControls.datum=datum   datum
    Line 8  DatumControls.FDir=floodDirection   flood direction
    Line 9  DatumControls.EDir=ebbDirection ebb direction
    Line 10     DatumControls.L2Flag=L2Flag L2Flag
    Line 11 DatumControls.HFlag=hydraulicFlag   hydraulic flag
    Line 12 DatumControls.RotFlag=0 For non-rotary tides, use “0”.  For rotary tides defined relative to North or East, use “1”.  For rotary tides defined by major and minor axes, use “2”.
    Lines 13-17 constituent amplitudes in order     M2, S2, N2, K1, M4, O1, M6, MK3, S4, MN4, NU2, S6, MU2, 2N2, OO1, LAM2, S1, M1, J1, MM, SSA, SA, MSF, MF, RHO, Q1, T2, R2, 2Q1, P1, 2SM2, M3, L2, 2MK3, K2, M8, MS4 [knots]
    Lines 18-23 constituent phases in order see above, lines 13-17 [degrees]
    Line 24 [Offset]
    Note:  Lines 25-38 use a second integer to indicate to GNOME whether there is valid data in the field.  “0” indicates no data, so GNOME can skip the calculation.  “1” indicates valid data exists (that may be zero).
    Line 25 MinBefFloodTime= minBefFloodTime 1  minimum before flood time adjustment
    Line 26 FloodTime= floodTime 1  flood time adjustment
    Line 27 MinBefEbbTime= minBefEbbTime 1  minimum before ebb time adjustment
    Line 28 EbbTime= ebbTime 1  ebb time adjustment
    Line 29 FloodSpdRatio=floodSpeedRatio 1 flood speed ratio
    Line 30 EbbSpdRatio=ebbSpeedRatio 1 ebb speed ratio
    Line 31 MinBFloodSpd=minBefFloodAvgSpeed 0  average speed - minimum before flood
    Line 32 MinBFloodDir=minBefFloodAvgDir 0    average direction - minimum before flood
    Line 33 MaxFloodSpd=maxFloodAvgSpeed 0  average speed - flood
    Line 34 MaxFloodDir=maxFloodAvgDir 0    average direction - flood
    Line 35 MinBEbbSpd=minBefEbbAvgSpeed 0  average speed - minimum before ebb
    Line 36 MinBEbbDir=minBefEbbAvgDir 0    average direction - minimum before ebb
    Line 37 MaxEbbSpd=maxEbbAvgSpeed 0  average speed – ebb
    Line 38 MaxEbbDir=maxEbbAvgDir 0    average direction – ebb

Example – Filename:  StJohnsRiver.shio.txt
..........................................

.. code-block:: none

    [StationInfo]
    Type=C
    Name=ST. JOHNS RIVER ENT. (between jetties)
    Latitude=30.400000
    Longitude=-81.383333
    [Constituents]
    DatumControls.datum=-0.350000
    DatumControls.FDir=275
    DatumControls.EDir=100
    DatumControls.L2Flag=0
    DatumControls.HFlag=0
    DatumControls.RotFlag=0
    H=1.993000 0.333000 0.404000 0.216000 0.293000 0.174000 0.092000 0.000000 0.000000 0.000000 0.078000 0.000000 0.000000 0.054000 0.000000 0.014000 0.000000 0.012000 0.014000 0.000000 0.000000 0.000000 0.000000 0.000000 0.000000 0.034000 0.020000 0.000000 0.000000 0.071000 0.000000 0.000000 0.054000 0.000000 0.091000 0.044000 0.000000 0.000000 0.000000 0.000000 0.000000 0.000000
    kPrime=227.199997 244.399994 208.800003 98.800003 131.100006 122.699997 238.699997 0.000000 0.000000 0.000000 211.199997 0.000000 0.000000 190.300003 0.000000 235.199997 0.000000 110.699997 86.800003 0.000000 0.000000 0.000000 0.000000 0.000000 0.000000 134.600006 244.600006 0.000000 0.000000 99.199997 0.000000 0.000000 245.699997 0.000000 244.000000 100.800003 0.000000 0.000000 0.000000 0.000000 0.000000 0.000000
    [Offset]
    MinBefFloodTime=0.000000 1
    FloodTime=0.000000 1
    MinBefEbbTime=0.000000 1
    EbbTime=0.000000 1
    FloodSpdRatio=1.000000 1
    EbbSpdRatio=1.000000 1
    MinBFloodSpd=0.000000 0
    MinBFloodDir=0.000000 0
    MaxFloodSpd=0.000000 0
    MaxFloodDir=0.000000 0
    MinBEbbSpd=0.000000 0
    MinBEbbDir=0.000000 0
    MaxEbbSpd=0.000000 0
    MaxEbbDir=0.000000 0
    Example – Filename:  Edmonds.shio.txt
    [StationInfo]
    Type=C
    Name=Edmonds, 2.7 miles WSW of
    Latitude=47.800000
    Longitude=-122.450000
    [Constituents]
    DatumControls.datum=-0.500000
    DatumControls.FDir=180
    DatumControls.EDir=5
    DatumControls.L2Flag=0
    DatumControls.HFlag=0
    DatumControls.RotFlag=0
    H=1.954000 0.460000 0.402000 0.847000 0.000000 0.421000 0.000000 0.000000 0.000000 0.000000 0.078000 0.000000 0.000000 0.054000 0.018000 0.013000 0.000000 0.030000 0.033000 0.000000 0.000000 0.000000 0.000000 0.000000 0.016000 0.081000 0.028000 0.000000 0.000000 0.280000 0.000000 0.000000 0.055000 0.000000 0.125000 0.000000 0.000000 0.000000 0.000000 0.000000 0.000000 0.000000
    kPrime=66.400002 84.099998 39.400002 72.500000 0.000000 66.199997 0.000000 0.000000 0.000000 0.000000 43.000000 0.000000 0.000000 12.300000 78.800003 74.599998 0.000000 69.300003 75.800003 0.000000 0.000000 0.000000 0.000000 0.000000 63.500000 63.000000 84.400002 0.000000 0.000000 73.199997 0.000000 0.000000 93.400002 0.000000 83.400002 0.000000 0.000000 0.000000 0.000000 0.000000 0.000000 0.000000
    [Offset]
    MinBefFloodTime=0.733333 1
    FloodTime=0.100000 1
    MinBefEbbTime=0.216667 1
    EbbTime=0.316667 1
    FloodSpdRatio=0.100000 1
    EbbSpdRatio=0.200000 1
    MinBFloodSpd=0.000000 1
    MinBFloodDir=0.000000 0
    MaxFloodSpd=0.200000 1
    MaxFloodDir=170.000000 1
    MinBEbbSpd=0.000000 1
    MinBEbbDir=0.000000 0
    MaxEbbSpd=0.500000 1
    MaxEbbDir=0.000000 1


Hydrology Time-Series
.....................

Hydrology time-series files for currents have the format per the bulleted list below.  An example hydrology time-series file, Hillsbourgh.HYD, is also provided.

 * The first line lists the station name.
 * The second line contains the reference point position for scaling the current pattern with the hydrology volume transport time-series.
 * The third line provides the units for the volume transport:
     * cubic feet per second (CFS)
     * kilo cubic feet per second (KCFS)
       Defined as 1,000 cubic feet of water passing a given point for an entire second.
     * cubic meters per second (CMS)
     * kilo cubic meters per second (KCMS)
       Defined as 1,000 cubic meters of water passing a given point for an entire second.

The data are given in the same time-series format as the currents, except that the magnitude of the current is changed to the volume transport.

.. rubric:: Example – Filename:  Hillsbourgh.HYD

.. code-block:: none

    HILLSBOURGH STATION
    28.029534,-82.688080
    CMS
    01,10,2002,0,0,432,0
    02,10,2002,0,0,309,0
    03,10,2002,0,0,310,0
    04,10,2002,0,0,312,0
    05,10,2002,0,0,311,0
    06,10,2002,0,0,287,0
    07,10,2002,0,0,234,0
    08,10,2002,0,0,235,0
    09,10,2002,0,0,232,0
    10,10,2002,0,0,177,0

Annotated Version of the File

.. code-block:: none

    HILLSBOURGH STATION     # Station Name
    28.029534,-82.688080    # Position (latitude, longitude)
    CMS Units
    #  Day,  Month,  Year,  Hour,  Min.,  Transport,  Direction (Dummy Value)
    01, 10, 2002,  0,  0,  432,  0
    02, 10, 2002,  0,  0,  309,  0
    03, 10, 2002,  0,  0,  310,  0
    04, 10, 2002,  0,  0,  312,  0
    05, 10, 2002,  0,  0,  311,  0

