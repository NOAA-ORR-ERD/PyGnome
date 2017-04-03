Movers
======

Processes that change the position of the LEs are termed "movers" in GNOME. These can include advection of the LEs due to winds and currents, 
diffusive movement of LEs due to unresolved turbulent flow fields, and prescribed behavior of the LEs (e.g. rise velocity of oil droplets 
or larval swimming.)

Some examples and common use cases are shown here. For complete documentation see :mod:`gnome.movers`

Wind movers
-----------

Wind movers are tied to a wind object in the Environment class which is described
more fully in :doc:`weatherers`.
For example to create a wind mover based on manually entered time series::

    from gnome.model import Model
    from gnome.environment import Wind
    from gnome.movers import WindMover
    from gnome.basic_types import datetime_value_2d
    import numpy as np
    from datetime import datetime, timedelta
    start_time = datetime(2004, 12, 31, 13, 0)
    series = np.zeros((2, ), dtype=datetime_value_2d)
    series[0] = (start_time, (5, 180))
    series[1] = (start_time + timedelta(hours=18), (5, 180))
    
    model = Model()
    wind = Wind(timeseries=series,units='m/s')
    w_mover = WindMover(wind)
    model.movers += w_mover
    
Some helper functions are available in :mod:`gnome.scripting` for creating wind movers. For example, to 
create a wind mover from a single point time series in a text file::

    from gnome.scripting import wind_mover_from_file
    w_mover = wind_mover_from_file('mywind.txt')
    model.movers += w_mover
    
The format of the text file is described in the GNOME file formats document available `here 
<http://response.restoration.noaa.gov/sites/default/files/GNOME_DataFormats.pdf>`_.
Briefly, it has 3 header lines, followed by comma seperated data. Alternatively, the An example is given here with
annotations in brackets at the end of the lines:

|   23NM W Cape Mohican AK *(Location name, can be blank)*
|   60.240000, -168.223000 *(Latitude,longitude, can be blank)*
|   knots *(Units, eg: knots,mph,kph,mps)*
|   14, 10, 2015, 10, 0, 16.00, 20 *(day, month, year, hour, minute, speed, direction)*
|   14, 10, 2015, 11, 0, 16.00, 20
|   14, 10, 2015, 12, 0, 16.00, 20
|   14, 10, 2015, 13, 0, 13.00, 20
|


Gridded wind movers
-------------------

A spatially variable gridded wind can also be used to move particles in GNOME. At present, this functionality 
does not extend to weathering (i.e. all weathering algorithms use a single point wind time series regardless of
the particle location. The use of spatially variable winds for weathering processes is currently under development.
Since the gridded wind effects transport only, a wind object is not added to the model Environment class.

To create a gridded wind mover, we use the GridWindMover class::

    from gnome.movers import GridWindMover

    w_mover = GridWindMover('mygridwind.nc')
    model.movers += w_mover
    
The supported netCDF file formats for gridded winds are described `here 
<http://response.restoration.noaa.gov/sites/default/files/GNOME_DataFormats.pdf>`_.

Current movers
--------------

An example of implementing a simple current mover with a uniform current was described in 
the Scripting :doc:`scripting_intro`. More commonly, currents used to move particles in GNOME originate 
from models on regular, curvilinear, or unstructured (triangular) grids. 
Regardless of the grid type, we use the GridCurrentMover class::

    from gnome.movers import GridCurrentMover
    
    c_mover = GridCurrentMover('mygridcurrent.nc')
    model.movers += c_mover
    
The supported netCDF file formats for gridded currents are described `here 
<http://response.restoration.noaa.gov/sites/default/files/GNOME_DataFormats.pdf>`_.

Random movers
-------------

Randoms movers can be added to simulate both horizontal and vertical turbulent motions. 
Diffusion coefficients can be explicity specified or default values will be used. For 
example::

    from gnome.movers import RandomMover, RandomVerticalMover
    
    random_mover = RandomMover(diffusion_coef=10,000) #in cm/s
    model.movers += random_mover
    
    random_vert_mover = RandomVerticalMover(vertical_diffusion_coef_above_ml=10,vertical_diffusion_coef_below_ml=0.2,\
    mixed_layer_depth=10) #diffusion coefficients in cm/s, MLD in meters
    model.movers += random_vert_mover

Rise velocity movers
--------------------

The rise velocity mover depends on parameters specified when setting up a subsurface spill. For example, in the 
:ref:`subsurface_plume` example, we initialized a spill with a droplet size distribution of 10-300 microns. If we add 
a rise velocity mover, the rise velocities will be calculated based on the droplet size for each particle and the density 
of the specified oil. Since this information is associated with the spill object, we only need to create and add a rise 
velocity mover as follows::

    from gnome.movers import RiseVelocityMover
    
    rise_vel_mover = RiseVelocityMover
    model.movers += rise_vel_mover

As noted in the :ref:`subsurface_plume` example, a distribution of rise velocities can also be explicitly specified 
when initializing the subsurface release. To make all particles have the same rise velocity, specify a uniform distribution 
with the same value for high and low parameters. Here's a complete example where all particles will have a 1 m/s rise velocity::
    
    from gnome.model import Model
    from datetime import datetime, timedelta
    from gnome.scripting import subsurface_plume_spill
    from gnome.utilities.distributions import UniformDistribution
    from gnome.movers import RiseVelocityMover
    
    start_time = datetime(2015, 1, 1, 0, 0)
    model = Model(start_time=start_time,
              duration=timedelta(days=3),
              time_step=60 * 15, #seconds
              )
    ud = UniformDistribution(1,1)
    spill = subsurface_plume_spill(num_elements=1000,
                                   start_position=(-144,48.5, -1000.0),
                                   release_time=start_time,
                                   distribution=ud,
                                   distribution_type='rise_velocity',
                                   end_release_time = start_time + timedelta(days=1),
                                   amount=5000,
                                   substance='ALASKA NORTH SLOPE (MIDDLE PIPELINE)',
                                   units='bbl',
                                   windage_range=(0.01,0.02),
                                   windage_persist=-1,
                                   name='My spill')
    model.spills += spill
    
    rise_vel_mover = RiseVelocityMover()
    model.movers += rise_vel_mover
    
    model.full_run()

PyMovers
----------

This new type of mover includes the gnome.environment.PyGridCurrentMover and gnome.environment.PyWindMover. They are 
being developed to work more seamlessly with native model grids (e.g. staggered grids) and will ultimately replace GridCurrentMover and GridWindMover. However, they are still under active development and this documentation may not
accurately reflect the current state of development.

PyMovers are built to work with the Property objects, and also provide multiple types of numerical methods for moving the particles. ::

    from gnome.environment.property_classes import GridCurrent
    from gnome.movers import PyGridCurrentMover
    fn = 'my_data.nc'
    current = GridCurrent.from_netCDF(filename=fn)
    curr_mover = PyGridCurrentMover(current)

There are three types of numerical methods currently supported.

1. Euler method ('Euler')
2. Trapezoidal/Heun's 2nd order method ('Trapezoid')
3. Runge-Kutta 4th order method ('RK4')

To use them, set the 'default_num_method' argument when constructing a mover. Alternatively, you may alter the mover as follows: ::

    fn = 'my_data.nc'
    current = GridCurrent.from_netCDF(filename=fn)
    curr_mover = PyGridCurrentMover(current, default_num_method = 'RK4')
    
    #RK4 is too slow, so lets go to the 2nd order method.
    curr_mover.default_num_method = 'Trapezoid'
    
The get_move function has the same interface as previous movers. You may also pass in a numerical method here and it will use it instead
of the default. ::

    curr_mover.get_move(sc, time_step, model_time_datetime, num_method = 'Euler')
    