.. include:: ../links.rst
Environment Objects
===================

Transport and weathering of particles in GNOME depend on a variety of environmental conditions (e.g. wind, waves, and water properties). Environment objects are used to represent this interface with an interface that can be queried in time and space.

Environnment objects can represent a space-independent time series or gridded, time dependent data. Regardless of the structure of the underlying data, the interface to access the information is identical as illustrated in the examples below.

For detailed documentation of the API and implemented objects see :mod:`gnome.environment.environment_objects`

.. note:: Environment Objects

    An important note is that environment objects alone do not have any effect on the model simulation. Once they are created, they can be explicitly passed to weatherers and movers. However, if a weatherer is added to the model without explicity specifying the required environment objects, then the first object of the correct type in the environment collection will be used for that weathering process.
    For example, if multiple wind time series are created and added to
    ``model.environment`` then the first one added will be used
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
    # Make a wind time series (one value for wind constant in time)
    series = np.zeros((1, ), dtype=datetime_value_2d)
    series[0] = (datetime(2015, 1, 1, 0), (10, 0))
    wind = gs.Wind(timeseries=series, units='knots')

This is still rather complicated. Much more simply, we can use the helper function for creating a constant wind::

    wind = gs.constant_wind(10,0,'knots')

Alternatively, if we had a properly formatted file (:ref:`wind_formats`) with a timeseries of wind data at a single point, we could use that to create a Wind Object using the Wind Class that is imported into the scripting module for convenience::

    wind = gs.Wind(filename='wind_file.txt')

Since the environment object does not act on the particles, it is not necessary to add it to the model. Instead, we use this object to create movers and  weatherers that are based on these objects.

Once the object is created, the information contained is accessed using the "at()" method.::

	wind_value = wind.at([-125.5,48,0],datetime.datetime.now())

In this case, the wind was constant in both space and time so I can query it anywhere at any time and get the same constant value of 10 knots. 

Gridded Environment Objects
---------------------------

The models set up with PyGNOME are often driven with data created by hydrodynamic and atmospheric models, such as ROMS, HYCOM, etc. Typically output from these models is netCDF data files. To create a GridWind environment object from a netCDF file::

    import gnome.scripting as gs

    fn = 'my_current_file.nc'
    wind = gs.GridWind.from_netCDF(filename=fn)

One major advantage to environment objects is re-use of common attributes. For example, in a data file you have a grid, and
wind and current variables that are associated with the grid. ::

    current = gs.GridCurrent.from_netCDF(filename=fn)
    wind = gs.GridWind.from_netCDF(filename=fn,
                                   grid=current.grid)

In the above example, the current and wind now both share the same grid object, which has numerous performance benefits. This is
one of the most common cases of sharing between Environment objects.

Ice Aware Objects
-----------------

For simulations including ice, there are several important environment objects that need to be created. This section is under construction.

:class:`gnome.environment.IceAwareCurrent`

:class:`gnome.environment.IceAwareWind`

:class:`gnome.environment.IceConcentration`

:class:`gnome.environment.IceDrift`


Other Environment Objects
-------------------------

This section is under construction.

:class:`gnome.environment.Water`

:class:`gnome.environment.Tide`

:class:`gnome.environment.Waves`

More examples of the interaction of environment objects with movers and weatherers will be given in the next section.





