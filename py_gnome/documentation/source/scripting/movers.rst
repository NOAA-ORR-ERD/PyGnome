Movers
======

Processes that change the position of the particles are termed "movers" in PyGNOME. These can include advection of the particles due to winds and currents, 
diffusive movement of particles due to unresolved (sub-grid scale) turbulent flow fields, and prescribed behavior of the particles (e.g. rise velocity of oil droplets).
Many of the movers derive their data from :ref:`scripting_environment`, which represent the environment in which the model is running.
The environment objects are queried for information (winds, currents) at the particle location in order to move the particle appropriately.

Some examples and common use cases are shown here. For comprehensive documentation see :mod:`gnome.movers` in the API Reference section.

Point Wind Mover
----------------

.. todo:: We should have a PointWind object which can be used with the regular Wind Mover, and then, maybe a PointWindMover that uses that.

A :class:`gnome.movers.PointWindMover` will act uniformly on elements anywhere in the domain (i.e. they have no spatial variability). These movers are tied to a Wind Object in the Environment Class.

For example, in that section, we saw how to create a simple spatially and temporally constant :class:`gnome.environment.Wind` using a helper function in the gnome scripting module::

    import gnome.scripting as gs
    model = gs.Model(start_time="2023-03-03",
                     duration=gs.days(1),
                     time_step=gs.minutes(15)
                     )
    wind = gs.constant_wind(10, 0, 'knots')
    
Now we create a :class:`gnome.movers.PointWindMover` by passing the Wind Object to the Mover Class and adding it to the model::

    w_mover = gs.PointWindMover(wind)
    model.movers += w_mover
    
Even though we didn't explicitly add the wind to the model environment, when the mover is added to the model, the wind object will also be added. Any weatherers subsequently added to the model will use that wind by default (see next section).

Some helper functions are available in :mod:`gnome.scripting` for creating wind movers.
Many of these helper functions automatically create and add environment objects to the model.
For example, to create a wind mover from a single point time series in a text file::

    w_mover = gs.point_wind_mover_from_file('wind_file.txt')
    model.movers += w_mover
    
The format of the text file is described in the :doc:`../file_formats/index` section.
Briefly, it has 3 header lines, followed by comma seperated data. An example is given here with annotations in brackets at the end of the lines:

.. code-block:: none

   22NM W Forks WA *(Location name, can be blank)*
   47.904000, -124.936000 *(Latitude, longitude, can be blank)*
   knots *(Units, eg: knots,mph,kph,mps)*
   3, 3, 2023, 12, 0, 14.00, 200 *(day, month, year, hour, minute, speed, direction)*
   3, 3, 2023, 13, 0, 16.00, 190
   3, 3, 2023, 14, 0, 16.00, 190

Gridded movers
--------------

An example of implementing a simple current mover with a uniform current was described in the scripting :doc:`scripting_intro`.
More commonly, currents used to move elements in GNOME originate
from models on regular, curvilinear, or unstructured (triangular) grids, as output from oceanographic or meteorological models.
Regardless of grid type, we use the :class:`gnome.movers.CurrentMover` class.

Similarly, winds can be derived from gridded meteorological models using the :class:`gnome.movers.WindMover` class.

These movers are tied to objects in the :class:`gnome.environment.Environment` which were described
more fully in the previous section. The primary supported format for gridded winds and currents is NetCDF. See the :doc:`../file_formats/netcdf` section for more information.

Here's an example of first building an environment object from a gridded wind::

    fn = 'gridded_wind.nc'
    wind = gs.GridWind.from_netCDF(filename=fn)
    wind_mover = gs.WindMover(wind)
    model.movers += wind_mover

The work flow is identical for adding a current. Alternatively, we could skip explicitly creating the environment object as the mover classes also have the "from_netCDF" method. For example::

    fn = 'gridded_current.nc'
    current_mover = gs.CurrentMover.from_netCDF(filename=fn)
    model.movers += current_mover
    
In both cases, the corresponding environment object is also added to the model.

The default numerical method for the gridded movers is a 2nd-order Runge-Kutta. Other options are available by specifying the "default_num_method" when creating the mover object. For more information, see the :class:`gnome.movers.CurrentMover` api documentation.

.. admonition:: A note on 3D simulations

    If a netCDF file contains currents at multiple depth levels along with 3-d grid information, the corresponding GridCurrent object will be built to include that information and full 3D simulations can be run. If only one depth level is included, it will be assumed to be the surface and used accordingly. Wind files should ideally only contain surface (assumed 10 m) winds. 
  
Random movers
-------------

Randoms movers can be added to simulate both horizontal and vertical turbulent motions (for 3d simulations). Diffusion coefficients can be explicitly specified or default values will be used. For example::

    import gnome.scripting as gs
       
    random_mover = gs.RandomMover(diffusion_coef=10000) #in cm/s
    model.movers += random_mover
    
    #Or, for  a 3D simulation
    random_mover_3d = gs.RandomMover3D(vertical_diffusion_coef_above_ml=10,vertical_diffusion_coef_below_ml=0.2,\
    mixed_layer_depth=10, horizontal_diffusion_coef_above_ml=10000,\
    horizontal_diffusion_coef_below_ml=100) #diffusion coefficients in cm/s, MLD in meters
    model.movers += random_mover_3d

Rise velocity movers
--------------------

The rise velocity mover depends on parameters specified when setting up a subsurface spill (see :doc:`spills`). For example, the rise velocities can be calculated based on the droplet size for each element and the density of the specified oil.
This information is associated with the spill object, hence creating a :class:`RiseVelocityMover` is relatively simple.::

    import gnome.scripting as gs
    
    rise_vel_mover = gs.RiseVelocityMover()
    model.movers += rise_vel_mover

A distribution of rise velocities can also be explicitly specified -- again this is done when initializing the subsurface release. To make all elements have the same rise velocity, we specify a uniform distribution with the same value for high and low parameters. Various distributions are available in :mod:`gnome.utilities.distributions`.

Here's a complete example where all elements will have a 1 m/s rise velocity::

    import gnome.scripting as gs
    from gnome.utilities.distributions import UniformDistribution

    start_time = gs.asdatetime("2023-03-03")
    model = gs.Model(start_time=start_time,
                     duration=gs.days(3),
                     time_step=60 * 15, #seconds
                     )
    ud = UniformDistribution(1,1)
    spill = gs.subsurface_spill(num_elements=1000,
                                start_position=(-144,48.5,-1000.0),
                                release_time=start_time,
                                distribution=ud,
                                distribution_type='rise_velocity',
                                end_release_time = start_time + gs.days(1),
                                amount=5000,
                                units='bbl',
                                name='My spill')
    model.spills += spill

    rise_vel_mover = gs.RiseVelocityMover()
    model.movers += rise_vel_mover

    model.full_run()

Ice modified movers
-------------------

The presence of ice modifies the movement of the oil on the water surface. For example, in high ice concentrations, the oil may be encapsulated in the ice, and move with the ice drift velocity. To incorporate the presence of ice requires the creation of environment objects that include the relevant information (e.g., ice concentration and ice velocity along with currents and winds). We term these "IceAware" environment objects (see previous section for more detail). Once the environment objects have been created, movers can be created based on them using the same approach described above. For example::

    ice_aware_current = gs.IceAwareCurrent.from_netCDF('file_with_currents_ice.nc')
    ice_current_mover = gs.CurrentMover(ice_aware_current)

CATS  Movers
------------

CATS is a NOAA/ORR hydrodynamic model that is unlikley to be used by others. Documentation forthcoming.