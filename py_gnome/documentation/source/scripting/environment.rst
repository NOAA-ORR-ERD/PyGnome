.. include:: ../links.rst

.. _scripting_environment:

Environment Objects
===================

Transport and weathering of particles in GNOME depend on a variety of environmental conditions (e.g. wind, waves, and water properties). Environment objects are used to represent this interface with an interface that can be queried in time and space.

Environnment objects can represent a space-independent time series or gridded, time dependent data. Regardless of the structure of the underlying data, the interface to access the information is identical as illustrated in the examples below.

For detailed documentation of the API and implemented objects see :mod:`gnome.environment.environment_objects`

.. note:: 

    An important note is that environment objects alone do not have any effect on the model simulation.
    Once they are created, they can be explicitly passed to weatherers and movers. However, if a weatherer is added to the model without explicity specifying the required environment objects, then the first object of the correct type in the environment collection will be used for that weathering process.
    For example, if multiple wind time series are created and added to
    ``model.environment`` then the first one added will be used
    for weathering processes unless explicitly specified.

Wind Objects
------------

``Wind`` objects represent the surface wind that affects the elements in the model: moving them via "windage", and influencing the evaporation and dispersion processes.

PyGNOME includes two types of ``Wind`` environment objects:
:class:`gnome.environment.wind.PointWind` and
:class:`gnome.environment.wind.environment_objects.GridWind`

``PointWind`` represents a time series of wind at a point, such as a single met station or forecast location. The same value is applied over all locations.

``GridWind`` provides gridded surface wind data, such as from a meteorological model, which can vary over time and space.

PointWind
.........

A ``PointWind`` can be created from a timeseries of wind data, or, more commonly, from a file in the "OSSM" format (see: :ref:`wind_formats`)::

    wind = gs.PointWind(filename='wind_file.txt')

Even more simply, there is utility function for creating a constant wind::

    wind = gs.constant_wind(10, 45, 'knots')

Which creates a constant wind with a speed of 10 knots, and a direction of 34 degrees -- from the northeast.

In order to create a wind time series from raw data, you can create a TimeSeries object first, and create the PointWind from that::

    import gnome.scripting as gs
    import numpy as np
    from datetime import datetime
    from gnome.basic_types import datetime_value_2d
    # Make a wind time series -- three values -- a short forecast
    series = np.zeros((3, ), dtype=datetime_value_2d)
    series[0] = (datetime(2015, 1, 1, 0), (10, 0))
    series[1] = (datetime(2015, 1, 1, 1), (12, 10))
    series[2] = (datetime(2015, 1, 1, 2), (15, 25))
    wind = gs.PointWind(timeseries=series, units='knots')

.. todo:: If we want to do that, we should probably make utility function to makme it easier.

Note: this Environment object represents the wind itself, it does not act on the elements. This object can be used as a "driver" for  movers and weatherers that affect the elements.

Once the PointWind object is created, the information contained is accessed using the "at()" method.::

	wind_value = wind.at([-125.5,48,0], datetime(2015, 1, 1, 1, 30))

The wind is constant in time so it will return the same value for any location. IN time, the value is interpolated to the time asked for. If queried outside the time series provided, it will raise an Error.

Extrapolation:
..............

By default, the PointWind object will not extrapolate beyond teh specified time series::

    In [13]: wind.extrapolation_is_allowed
    Out[13]: False

But you can set that to True, and then it will return the end value when extrapolated:

    In [15]: wind.extrapolation_is_allowed = True

    In [17]: wind_value = wind.at([-125.5,48,0], datetime.now())

    In [18]: wind_value
    Out[18]: array([[-3.26120144, -6.99366905,  0.        ]])


Gridded Environment Objects
---------------------------

The models set up with PyGNOME are often driven with data created by hydrodynamic and atmospheric models, such as ROMS, HYCOM, etc. The most common output format from these models is the NetCDF file format. To create a :mod:`gnome.environment.GridWind` environment object from a NetCDF file::

    import gnome.scripting as gs

    fn = 'my_data_file.nc'
    wind = gs.GridWind.from_netCDF(filename=fn)

One major advantage of environment objects is re-use of common attributes. For example, if you have a data file with 
wind and current variables that are associated with the same grid. ::

    current = gs.GridCurrent.from_netCDF(filename=fn)
    wind = gs.GridWind.from_netCDF(filename=fn,
                                   grid=current.grid)

In the above example, the current and wind both share the same grid object, which has numerous performance benefits. This is
one of the most common cases of sharing between Environment objects.

You can also create an environment out of an already open dataset. This may help alleviate 'too many files' problems when working
with large numbers of files::

    df = netCDF4.Dataset(fn)
    current = GridCurrent.from_netCDF(dataset=df)

Ice Aware Objects
-----------------

For simulations including ice, there are several important environment objects that need to be created. These objects require that the model output include ice concentration and ice drift velocity. The :mod:`IceAwareCurrent` and :mod:`IceAwareWind` are GridCurrent/GridWind instances that modulate the usual water velocity field depending on ice concentration. 

For an :class:`gnome.environment.IceAwareCurrent`::

    * While under 20% ice coverage, queries will return water velocity as in the non ice case.
    * Between 20% and 80% coverage, queries will interpolate linearly between water and ice drift velocity
    * Above 80% coverage, queries will return the ice drift velocity.

For an :class:`gnome.environment.IceAwareWind`::
    
    * While under 20% ice coverage, queries will return the wind velocity.
    * Between 20% and 80% coverage, queries will interpolate linearly between wind magnitude and zero.
    * Above 80% coverage, queries will return a wind magnitude of 0.

The following example shows how to create "ice aware" current and wind environment objects:: 

    fcurr = 'current_ice_file.nc'
    fwind = 'wind_ice_file.nc'
    ice_aware_curr = gs.IceAwareCurrent.from_netCDF(filename=fcurr)
    ice_aware_wind = gs.IceAwareCurrent.from_netCDF(filename=fwind)

If, as is common, the ice concentration and velocity data are only present in the currents netCDF file, to set-up an IceAwareWind object we will need information from the IceAwareCurrent object::

    ice_aware_wind = gs.IceAwareWind.from_netCDF(filename=fwind,
                            ice_concentration=ice_aware_curr.ice_concentration,
                            ice_velocity=ice_aware_curr.ice_velocity
                            )    


:class:`gnome.environment.IceConcentration`

:class:`gnome.environment.IceDrift`


Other Environment Objects
-------------------------

This section is under construction.

:class:`gnome.environment.Water`

:class:`gnome.environment.Tide`

:class:`gnome.environment.Waves`

More examples of the interaction of environment objects with movers and weatherers will be given in the next section.





