.. _WCOFS_netcdf:

WCOFS NetCDF formats
======================

This document provides information on the ROMS output variables required and/or used in 2D and 3D GNOME model applications.  It's organized in three sections to highlight the required outputs for:
- different purposes (e.g. advection and weathering)
- 2D applications
- 3D applications

ROMS output requirements by purpose/utility
-------------

**Advection/transport (required for all cases)**

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
-------------

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
-------------

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

