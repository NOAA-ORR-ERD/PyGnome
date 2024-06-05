.. _ROMS_netcdf:

ROMS Model: NetCDF output
=========================

This document provides information on the ROMS output variables required and/or used in 2D and 3D GNOME model applications.  It's organized in three sections to highlight the required outputs for:
- different purposes (e.g. advection and weathering)
- 2D applications
- 3D applications


ROMS output requirements by purpose/utility
-------------------------------------------

ROMS uses a "Staggered Grid", and PYGNOME seeks to be compliant with the SGRID conversion:

https://sgrid.github.io/sgrid/

However, many ROMS output files are not compliant with that convension,
so PyGNOME is designed to read ROMS specific files.
But the concepts are helpful, so ROMS has been described in that context here.

Grid definition
...............

ROMS is an Arakawa C-grid model -- the full grid definition is required in order to properly interpret the results.

**Required grid variables:**


.. +------------------------+------------+----------+----------+
.. | Header row, column 1   | Header 2   | Header 3 | Header 4 |
.. | (header rows optional) |            |          |          |
.. +========================+============+==========+==========+
.. | body row 1, column 1   | column 2   | column 3 | column 4 |
.. +------------------------+------------+----------+----------+
.. | body row 2             | ...        | ...      |          |
.. +------------------------+------------+----------+----------+

+-----------------------+----------------+---------------------------------------+
| CF/SGRID concept      | ROMS name      | Usual ROMS Variables                  |
+=======================+================+=======================================+
| node_coordinates      | psi points     | ``lon_psi``, ``lat_psi``, ``mask_psi``|
+-----------------------+----------------+---------------------------------------+
| face_coordinate       | rho points     | ``lon_rho``, ``lat_rho``, ``mask_rho``|
+-----------------------+----------------+---------------------------------------+
|          time         |  time          |         ``time`` or ``ocean_time``    |
+-----------------------+----------------+---------------------------------------+
| **Three dimensions**  |                |                                       |
+-----------------------+----------------+---------------------------------------+
|           depth       |      ???       |                                       |
+-----------------------+----------------+---------------------------------------+
|                       |                |                                       |
+-----------------------+----------------+---------------------------------------+
|                       |                |                                       |
+-----------------------+----------------+---------------------------------------+


.. note:: ROMS output often includes the 'u' and 'v' locations: the cneters of the cell edges. These are not used by PyGNOME **IS THIS TRUE??**


**Advection/transport**

  lon-psi

  lat-psi
  
  angle (necessary unless grid perfectly aligned with N/S, E/W).  No error is thrown but the results won't be right. 
  
  u (horizontal movers)
  
  v
  
  ocean_time

**Advection/transport (optional)**

  wet/dry 
  
  masks (this will affect code in ways that are a bit "off-roading")
  
  vertical diffusivities (vertical random mover)

**Advection/transport (required for 3D cases)**

  w (vertical mover)

  stretching params:
       		1. Cs_r,
    		2. Cs_w,
      		3. s_rho,
        	4. S_w,
         	5. hc,
          	6. Vtransform,
          	7. zeta (used to calculate sigma-levels), and
          	8. h

**Weathering**

  temp

  salt

  u10, eastward_wind, or wind_u

  v10, westward_wind, or wind_v

  [**future**] zeta (for 3D applications but not currently used)

  [**future**] sediment

**Visualization**

  lon-rho, -u, -v, -psi

  lat-rho, -u, -v, -psi

  mask-rho, -u, -v, -psi

2D case WCOFS output requirements 
---------------------------------

**Current transport file**

  ocean_time

  u 

  v

  horizontal diffusivities (random movers)

  temp

  salt

**The following 4 list items can be in a separate grid file or included in the current transport file**

  lon-rho, -u, -v, -psi (only psi required to run model.  The rest are for graphical display)

  lat-rho, -u, -v, -psi

  mask-rho, -u, -v, -psi

  angle (necessary unless grid perfectly aligned with N/S, E/W).  No error is thrown but the results won't be right. 

**wind forcing file**

  u10, eastward_wind, or wind_u

  v10, westward_wind, or wind_v

3D case WCOFS output requirements
---------------------------------

**Current transport file**

  ocean_time

  u 

  v

  w (vertical mover)

  vertical/horizontal diffusivities (random movers)

  temp

  salt

**The following 5 list items (through all stretching params) can be in a separate grid file or together with current transport file**
  
  lon-rho, -u, -v, -psi (only psi required to run model.  The rest are for graphical display)

  lat-rho, -u, -v, -psi

  mask-rho, -u, -v, -psi

  angle (necessary unless grid perfectly aligned with N/S, E/W).  No error is thrown but the results won't be right.

  stretching params:
   1. Cs_r,
   2. Cs_w,
   3. s_rho,
   4. S_w,
   5. hc,
   6. Vtransform,
   7. zeta (used to calculate sigma-levels), and
   8. h

**winds forcing file**

  u10, eastward_wind, or wind_u

  v10, westward_wind, or wind_v

.. NOTE: this was auto-built into the docstring of the:
..       gnome/environment/names.py file -- it would be
..       nice to auto-update, but this is start

.. and we should be able to link to the docstring ...

