.. include:: ../links.rst
Environment Objects
===================

As transport and weathering of particles in PyGNOME depend on a variety of environmental conditions (e.g. wind, waves, and water properties),
initialization of various environment objects is needed before these processes can be added to the model. 
Environment objects are designed to represent a natural phenomenon and provide an interface that can be queried in time and space. 

These objects
can represent a space-independent time series or gridded, time dependent data. They can also represent scalar and (2D) vector phenomena.
They are the core means for representing natural phenomena in PyGNOME. Examples of things that environment objects can represent include: temperature, water velocity, wind speed & direction time series, etc.

An environment object implements an association between a data variable (such as a netCDF Variable, or numpy array) and a
Grid, Time, and Depth (representing the data dimensions in space and time) and does interpolation across them.  In addition, if possible, the Grid, Time, and Depth may be shared among environment objects, which provides a number of performance and programmatic benefits.
The core functionality of an environment object is it’s ‘EnvObject.at(points, time)’ function, which provides an interface that allows the environment data to be intepolated to needed times and locations. 


..  In some cases, these objects may be automatically created. For    
    example,when creating a gridded 
    current mover from a NetCDF file (next section). Sometimes, it is necessary or desirable to manually
    create these objects. For example, if weatherering and transport processes may be dependent on the 
    same environmental information (winds) or if you want to enter data manually.


For detailed documentation of the API and implemented objects see :mod:`gnome.environment.environment_objects`

.. admonition:: Environment Objects

    An important note, is that environment objects alone do not have any effect on the model simulation. Once they are created, they can be explicitly passed to weatherers and movers. However, if a weatherer is added to the model without explicity specifying the required environment objects, then the first object of the correct type in the environment collection will be used for that weathering process. 
    For example, if multiple wind time series are created and added to model.environment then the first one added will be used
    for weathering processes unless explicitly specified.

Wind Object
-----------

Here's a detailed example to create a simple Wind object (for a constant in time wind). We'll take advantage of the gnome scripting 
module to avoid having to manually import the necessary classes and functions::

    import gnome.scripting as gs
    import numpy as np
    from datetime import datetime
    from gnome.basic_types import datetime_value_2d
    model = gs.Model(start_time="2015-01-01",
             duration=gs.days(3),
             time_step=gs.minutes(15)
             )
    series = np.zeros((1, ),dtype=datetime_value_2d) #Make a wind time series (one value for wind constant in time)
    series[0] = (datetime(2015,1,1,0), (10, 0))
    wind = gs.Wind(timeseries=series,units='knots')

This is still rather complicated. Much more simply, we can use the helper function for creating a constant wind::

    wind = gs.constant_wind(10,0,'knots')
    
Alternatively, if we had a properly formatted file (|file_formats_doc|) with a timeseries of wind data at a single point, we could use that to create a Wind Object using the Wind Class that is imported into the scripting module for convenience. An example file that was used for this ::

    wind = gs.Wind(filename='wind_file.txt')

Example of adding a manually adding a wind object to the model enviornment::

    model = gs.Model()
    model.environment += wind

Gridded Environment Objects
---------------------------

The models set up with pyGNOME are often driven with data created by other hydrodynamic and atmospheric models, such as ROMS, HYCOM, etc., and typical output from these models is netCDF data files. To create a GridWind environment object from a netCDF file::

    import gnome.scripting as gs

    fn = ('my_current_file.nc')
    wind = GridWind.from_netCDF(filename = fn)

One major advantage to environment objects is re-use of common attributes. For example, in a data file you have a grid, and
wind and current variables that are associated with the grid. ::

    current = GridCurrent.from_netCDF(filename = fn)
    wind = GridWind.from_netCDF(filename = fn,
                                grid = current.grid)

In the above example, the current and wind now both share the same grid object, which has numerous performance benefits. This is
one of the most common cases of sharing between Environment objects.

Ice Aware Objects
-----------------

For simulations including ice, there are several important environment objects that need to be created. This section is under construction.

:class:`gnome.environment.IceAwareCurrent'
:class:`gnome.environment.IceAwareWind'
:class:`gnome.environment.IceConcentration'
:class:`gnome.environment.IceDrift'


Other Environment Objects
-------------------------

This section is under construction.

:class:`gnome.environment.Water`
:class:`gnome.environment.Tide` 
:class:`gnome.environment.Waves` 


    
More examples of the interaction of environment objects with movers and weatherers will be given in the next section.





