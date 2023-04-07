.. _netcdf_formats:

Gridded NetCDF formats
======================

Ocean circulation and meteorological models output their results in a number of different grid formats. The goal of GNOME is to work with the native output for each model, and to do appropriate interpolation on the native grid. This assures fidelity to the original model, particularly near the boundaries, which are critical to oil spill applications.


Standards Compliant netCDF:
---------------------------

GNOME is designed to be able to directly read netcdf files that conform to the following standards:

The `CF Metadata Conventions <https://cfconventions.org/>`_
   These specify ways of storing data, standard names, etc, and are suitable for rectangular gridded model results.

The `Unstructured grid (UGRID) conventions <http://ugrid-conventions.github.io/ugrid-conventions/>`_
  These specify additions to the CF conventions for model results on  unstructured grids, in particular triangle meshes used in many finite element and finite volume models.

The `Staggered (SGRID) conventions <http://sgrid.github.io/sgrid/>`_
  These specify additions to the CF conventions for model results on  quadrilateral curvilinear grids (e.g. `Arakawa grids <https://en.wikipedia.org/wiki/Arakawa_grids>`_) used in many finite difference and hybrid models.

If your input file conform to these conventions, and use the standard names that GNOME is expecting, they should "just work". In particular, GNOME has known to work with the output from the following models:

- ROMS
- POM
- HYCOM
- FVCOM
- ADCRIC
- NOAA NAM
- NOAA GFS
- NEMO

If your files are compliant with the above standards and don't work with GNOME, it is a bug (or missing feature), and we would appreciate your letting us know via posting an issue on the `PyGNOME gitHub project <https://github.com/NOAA-ORR-ERD/PyGnome>`_

Standard Names expected by GNOME:
.................................

It is best to use the CF standard names with files used with GNOME. However, there are a lot of files in the wild that do not use the standard names, so GNOME will also look for a certain set of common variable names used for various parameters.

.. NOTE: this was auto-built into the docstring of the:
..       gnome/environment/names.py file -- it would be
..       nice to auto-update, but this is start

.. and we should be able to link to the docstring ...

Name Mapping:
-------------

**grid_temperature**

  Default Names: water_t, temp


  CF Standard Names: sea_water_temperature, sea_surface_temperature


**grid_salinity**

  Default Names: salt


  CF Standard Names: sea_water_salinity, sea_surface_salinity


**grid_sediment**

  Default Names: sand_06


  CF Standard Names:


**ice_concentration**

  Default Names: ice_fraction, aice


  CF Standard Names: sea_ice_area_fraction


**bathymetry**

  Default Names: h


  CF Standard Names: depth


**grid_current**

 Default Names for u: u, U, water_u, curr_ucmp, u_surface, u_sur

 Default Names for v: v, V, water_v, curr_vcmp, v_surface, v_sur

 Default Names for w: w, W


 CF Standard Names for u: eastward_sea_water_velocity, surface_eastward_sea_water_velocity

 CF Standard Names for v: northward_sea_water_velocity, surface_northward_sea_water_velocity

 CF Standard Names for w: upward_sea_water_velocity


**grid_wind**

 Default Names for u: air_u, Air_U, air_ucmp, wind_u, u-component_of_wind_height_above_ground

 Default Names for v: air_v, Air_V, air_vcmp, wind_v, v-component_of_wind_height_above_ground


 CF Standard Names for u: eastward_wind, eastward wind

 CF Standard Names for v: northward_wind, northward wind


**ice_velocity**

 Default Names for u: ice_u, uice

 Default Names for v: ice_v, vice


 CF Standard Names for u: eastward_sea_ice_velocity

 CF Standard Names for v: northward_sea_ice_velocity

Legacy netcdf
-------------

Early versions of GNOME supported netcdf before the above standards had been developed. As a result, there is a legacy "GNOME format" which is still supported by PyGNOME. It is better to use the above standards, but this should work.


GNOME can read in NetCDF files for rectangular, curvilinear, and triangular grids. This section includes examples of the three formats currently in use and some descriptions of the required information. Please note that the NetCDF formats described here are presently undergoing revision to conform to the newly forming Climate & Forecast unstructured grid data model, to be adopted in future releases of GNOME.

NetCDF Rectangular Grid
.......................

Below is an example of the regular grid format for NetCDF files. The global attribute grid_type = REGULAR is the default. Time units can be hours, minutes, seconds, or days. A separate map will be needed in order to set a spill.

.. code-block:: none

    NetCDF MacintoshHD:Desktop Folder:test {
    dimensions:
    lat = 16 ;
    lon = 20 ;
    time = UNLIMITED ;  (85 currently)
    variables:
    double lat(lat) ;
    lat:long_name = "Latitude" ;
    lat:units = "degrees_north" ;
    lat:point_spacing = "even" ;
    double lon(lon) ;
    lon:long_name = "Longitude" ;
    lon:units = "degrees_east" ;
    lon:point_spacing = "even" ;
    double time(time) ;
    time:long_name = "Valid Time" ;
    time:units = "minutes since 1999-11-25 00:00:00" ;
    float water_u(time, lat, lon) ;
    water_u:long_name = "Eastward Water Velocity" ;
    water_u:units = "m/s" ;
    water_u:_FillValue = -9.9999e+32f ;
    water_u:scale_factor = 1.f ;
    water_u:add_offset = 0.f ;
    float water_v(time, lat, lon) ;
    water_v:long_name = "Northward Water Velocity" ;
    water_v:units = "m/s" ;
    water_v:_FillValue = -9.9999e+32f ;
    water_v:scale_factor = 1.f ;
    water_v:add_offset = 0.f ;

    global attributes:
    :grid_type = "REGULAR" ;
    data:

    lat = 51.144606, 51.234386, 51.324167, 51.413944, 51.503722, 51.5935,
    51.683275, 51.77305, 51.862825, 51.952594, 52.042364, 52.132133, 52.2219,
    52.311664, 52.401425, 52.491186 ;

    lon = 2.3155722, 2.4583139, 2.6010833, 2.743875, 2.8866917, 3.0295306,
    3.1723917, 3.3152694, 3.4581667, 3.6010833, 3.7440139, 3.8869583,
    4.0299167, 4.1728861, 4.3158667, 4.4588583, 4.6018583, 4.7448639,
    4.887875, 5.0308917 ;

    time = 7020, 7080, 7140, 7200, 7260, 7320, 7380, 7440, 7500, 7560, 7620,
    7680, 7740, 7800, 7860, 7920, 7980, 8040, 8100, 8160, 8220, 8280, 8340,
    8400, 8460, 8520, 8580, 8640, 8700, 8760, 8820, 8880, 8940, 9000, 9060,
    9120, 9180, 9240, 9300, 9360, 9420, 9480, 9540, 9600, 9660, 9720, 9780,
    9840, 9900, 9960, 10020, 10080, 10140, 10200, 10260, 10320, 10380, 10440,
    10500, 10560, 10620, 10680, 10740, 10800, 10860, 10920, 10980, 11040,
    11100, 11160, 11220, 11280, 11340, 11400, 11460, 11520, 11580, 11640,
    11700, 11760, 11820, 11880, 11940, 12000, 12060 ;

NetCDF Curvilinear Grid
.......................

Below is an example of the curvilinear format for NetCDF files. The global attribute grid_type = CURVILINEAR is required (the default is grid_type = REGULAR). In addition to x and y, there are several other dimension name options for latitude and longitude. The dimension names only need to start with X, Y or LAT, LON to be recognized. The variable names must appear as shown. The velocities can be short, float, or double precision numbers. Time units can be hours, minutes, seconds, or days. The land-mask is required if you want to use the grid boundary as the shoreline: 0 is land, 1 is water. If no map is available, the mask is used to identify land points (land = 0, water = 1) and a boundary map is created. The first sigma value is used, though currently GNOME is being extended to handle 3-D currents. The topology can be saved out the first time and reloaded.

.. code-block:: none

    netcdf 20040726_11z_HAZMAT {
    dimensions:
    x = 73 ;
    y = 163 ;
    sigma = 3 ; optional
    time = UNLIMITED ;  (12 currently)
    variables:
    float time(time) ;
    time:long_name = "Time" ;
    time:base_date = 2004, 1, 1, 0 ;
    time:units = "days since 2004-01-01  0:00:00 00:00" ;
    time:standard_name = "time" ;
    float lon(y, x) ;
    lon:long_name = "Longitude" ;
    lon:units = "degrees_east" ;
    lon:standard_name = "longitude" ;
    float lat(y, x) ;
    lat:long_name = "Latitude" ;
    lat:units = "degrees_north" ;
    lat:standard_name = "latitude" ;
    float mask(y, x) ;
    mask:long_name = "Land Mask" ;
    mask:units = "nondimensional" ;
    float depth(y, x) ;     optional
    depth:long_name = "Bathymetry" ;
    depth:units = "meters" ;
    depth:positive = "down" ;
    depth:standard_name = "depth" ;
    float sigma(sigma) ;    optional
    sigma:long_name = "Sigma Stretched Vertical Coordinate at Nodes" ;
    sigma:units = "sigma_level" ;
    sigma:positive = "down" ;
    sigma:standard_name = "ocean_sigma_coordinate" ;
    sigma:formula_terms = "sigma: sigma eta: zeta depth: depth" ;
    float u(time, sigma, y, x) ;
    u:long_name = "Eastward Water Velocity" ;
    u:units = "m/s" ;
    u:missing_value = -99999.f ;
    u:_FillValue = -99999.f ;
    u:standard_name = "eastward_sea_water_velocity" ;
    float v(time, sigma, y, x) ;
    v:long_name = "Northward Water Velocity" ;
    v:units = "m/s" ;
    v:missing_value = -99999.f ;
    v:_FillValue = -99999.f ;
    v:standard_name = "northward_sea_water_velocity" ;

    global attributes:
    :file_type = "Full_Grid" ;
    :Conventions = "COARDS" ;
    :grid_type = "curvilinear" ;
    :z_type = "sigma" ;
    :model = "POM" ;
    :title = "Forecast: wind+tide+river" ;
    data:

    time = 208.4688, 208.4792, 208.4896, 208.5, 208.5104, 208.5208, 208.5312,
    208.5417, 208.5521, 208.5625, 208.5729, 208.5833,,;

    sigma = 0, .5, 1.;
    }

NetCDF Triangular Grid
......................

.. rubric:: Example – Triangular Grid Format with Velocities on the Nodes

Below is an example of the triangular grid format for NetCDF files with velocities on the nodes. The global attribute grid_type = TRIANGULAR is required (the default is grid_type = REGULAR). The first depth value is used. Time units can be hours, minutes, seconds, or days. A map will be created using the boundary data. The topology can be saved out the first time and reloaded.
The NetCDF header description for finite element model:

.. code-block:: none

    NetCDF MacintoshHD:Desktop Folder:testFile {
    dimensions:
    node = 7258 ;
    nele = 13044 ;  not currently used
    nbnd = 1476 ;
    nbi = 4 ;
    sigma = 11 ;    optional
    time = UNLIMITED ;  (12 currently)
    variables:
    short bnd(nbnd, nbi) ;
    bnd:long_name = "Boundary Segment Node List" ;
    bnd:units = "index_start_1" ;
    float time(time) ;
    time:long_name = "Time" ;
    time:units = "days since 2003-01-00  0:00:00 00:00" ;
    time:base_date = 2003, 1, 0, 0 ;
    float lon(node) ;
    lon:long_name = "Longitude" ;
    lon:units = "degrees_east" ;
    float lat(node) ;
    lat:long_name = "Latitude" ;
    lat:units = "degrees_north" ;
    float sigma(sigma) ;    optional
    sigma:long_name = "Stretched Vertical Coordinate" ;
    sigma:units = "sigma_level" ;
    sigma:positive = "down" ;
    float u(time, sigma, node) ;
    u:long_name = "Eastward Water Velocity" ;
    u:units = "m/s" ;
    u:missing_value = -99999.f ;
    u:_FillValue = -99999.f ;
    float v(time, sigma, node) ;
    v:long_name = "Northward Water Velocity" ;
    v:units = "m/s" ;
    v:missing_value = -99999.f ;
    v:_FillValue = -99999.f ;

    global attributes:
    :file_type = "FEM" ;
    :Conventions = "COARDS" ;
    :grid_type = "Triangular" ;
    data:

    time = 26.95833, 27, 27.04167, 27.08333, 27.125, 27.16667, 27.20833, 27.25,
    27.29167, 27.33333, 27.375, 27.41667 ;

    sigma = 1, 0.9807215, 0.9306101, 0.83061, 0.6807215, 0.5, 0.3192785,
    0.1693899, 0.06938996, 0.01927857, 0 ;
    }
    Notes:
    1.  The boundary list is an array of dimension bnd(nbnd, 4). It consists of node numbers of the line segments, with a digit to indicate which land or island the segment is a part of, and a digit to indicate whether a boundary is land or water:
    node1   node2   island  land/water (0/1)
    1   2   0   0   1 is usually the continent and outer water BC
    2   5   0   0
    5   23  0   1
    …
    3568    1   0   1   The last segment joins up with the first.
      551   552 1   0   next island
      552   567 1   0
    …
      677   551 1   0
      789   388 2   0
    …               next island, etc.
    2.  Only the first sigma level is used, although GNOME is currently being extended to handle 3-D currents.
     
    1.2.2.3.2   Example – Triangular Grid Format with Velocities on the Triangles
    Following is an example of the triangular grid format for NetCDF files with velocities on the triangles. The global attribute grid_type = TRIANGULAR is required (the default is grid_type = REGULAR). The first depth value is used. Time units can be hours, minutes, seconds, or days. A map will be created using the boundary data. The topology must be included in the file.
    netcdf FVCOM_example {
    dimensions:
    node = 32649 ;
    nele = 60213 ;
    nbnd = 5099 ;
    nbi = 4 ;
    time = UNLIMITED ; // (1 currently)
    three = 3 ;
    variables:
    int bnd(nbnd, nbi) ;
    float time(time) ;
    time:units = "days since 1978-11-17 00:00:00 0:00" ;
    time:long_name = "time" ;
    time:time_zone = "UTC" ;
    time:format = "modified julian day (MJD)" ;
    float lon(node) ;
    float lat(node) ;
    float u(time, nele) ;
    u:units = "meters s-1" ;
    u:long_name = "Eastward Water Velocity" ;
    u:grid = "fvcom_grid" ;
    u:type = "data" ;
    float v(time, nele) ;
    v:units = "meters s-1" ;
    v:long_name = "Northward Water Velocity" ;
    v:grid = "fvcom_grid" ;
    v:type = "data" ;
    int nbe(three, nele) ;
    int nv(three, nele) ;

    // global attributes:
    :grid_type = "Triangular" ;
    data:

    time = 11452 ;
    }

**Notes:**

1.  The boundary list is an array of dimension bnd(nbnd, 4), same as above.

2.  The triangle vertices are contained in nv and the neighboring triangles in nbe.


.. rubric:: Data in Multiple NetCDF Files:  When Your NetCDF Files Start To Get Too Big

.. note:: This approach is deprecated -- in current versions, you can simply pass the list of filenames to PyGNOME. [How is it done in WebGNOME?]


Longer simulations require more model data, and that can cause problems with putting the entire time-series into one data file. GNOME allows you to break the time-series into separate files using a master file to identify all the pieces of the time-series in order. This also makes possible using a series of nowcasts and forecasts strung together to make a times-series. This technique worked well during the 2002 T/V Prestige incident in Spain.

First create a text master file with the list of file path-names (relative to the GNOME directory) in order. Next supply the full path name if the files are not in the same directory as GNOME, or in a subdirectory. The file will also need a header line, “NetCDF Files”.
When you go to load the currents in GNOME, load your master file (e.g.,

.. rubric:: Example 1 – Filename: MyMasterFileEx.txt

.. code-block:: none

    NetCDF Files
    [FILE]  :day1.nc
    [FILE]  :day2.nc
    [FILE]  :day3.nc
    [FILE]  :day4.nc
    [FILE]  :day5.nc
    [FILE]  :day6.nc





