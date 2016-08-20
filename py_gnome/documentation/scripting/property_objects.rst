Property Objects
================

Property objects are designed to represent a natural phenomenon and provide an interface that can be queried in time and space. These objects
can represent a space-independent time series or gridded, time dependent data. They can also represent scalar and (2D) vector phenomena.
They are the core means for representing natural phenomena in the latest versions of PyGNOME.

Examples of things that Property objects can represent include: temperature, water velocity, wind speed & direction time series, etc.

For documentation of the API see :mod:`gnome.environment.property`
For documentation of implemented Properties, see :mod:`gnome.environment.property_classes`


Background
----------

The environmental models in GNOME are often driven with data created by other models, such as ROMS, HYCOM, etc. In the past, the output from
these models were processed by renaming and regridding to conform to GNOME's expectations before use.

Examples
--------

Create GridCurrent from a netCDF file.::

    import numpy as np
    import netCDF4 as nc4
    from datetime import datetime, timedelta
    from gnome.environment.property_classes import GridCurrent

    fn = ('my_current_file.nc')
    current = GridCurrent.from_netCDF(filename = fn)

Create GridCurrent from a netCDF file, and specify grid topology.::

    current = GridCurrent.from_netCDF(filename=fn,
                                      grid_topology={'node_lon':'lon',
                                                     'node_lat':'lat'}
                                      )

Again, this time specifying variable names as well.::

    current = GridCurrent.from_netCDF(filename=fn,
                                      grid_topology={'node_lon':'lon',
                                                     'node_lat':'lat'},
                                      varnames=['water_u', 'water_v']
                                      )

One major advantage to Property objects is re-use of common attributes. For example, in a data file you have a grid, and
wind and current variables that are associated with the grid. ::

    current = GridCurrent.from_netCDF(filename = fn)
    wind = GridWind.from_netCDF(filename = fn,
                                grid = current.grid)

In the above example, the current and wind now both share the same grid object, which has numerous performance benefits. This is
one of the most common cases of sharing between Property objects.

You can create a Property out of an already open dataset as well. This may help alleviate 'too many files' problems when working
with large numbers of files::

    df = netCDF4.Dataset(fn)
    current = GridCurrent.from_netCDF(dataset=df)

You can also set up a Property object from scratch, which can be very useful when mocking up a situation. The following code creates
a GridCurrent representing circular currents around the origin.::

    #create the rectangular grid
    x, y = np.mgrid[-30:30:61j, -30:30:61j]
    y = np.ascontiguousarray(y.T)
    x = np.ascontiguousarray(x.T)
    g = SGrid(node_lon=x,
              node_lat=y)
    g.build_celltree()

    #using a single time implies extrapolation
    t = datetime(2000,1,1,0,0)

    #create the data associated with each grid vertex
    angs = -np.arctan2(y,x)
    mag = np.sqrt(x**2 + y**2)
    vx = np.cos(angs) * mag
    vy = np.sin(angs) * mag
    vx = vx[np.newaxis,:] * 20
    vy = vy[np.newaxis,:] * 20

    #Create Property objects for the components separately, then combine into a GridCurrent
    vels_x = GriddedProp(name='v_x',units='m/s',time=[t], grid=g, data=vx)
    vels_y = GriddedProp(name='v_y',units='m/s',time=[t], grid=g, data=vy)
    vg = GridCurrent(variables = [vels_y, vels_x], time=[t], grid=g, units='m/s')

Defining a new Property
------------------------

To create a new Property, let us take the example of water temperature.

1. It is scalar, so it would inherit from a scalar-type Property
2. In this example, it is gridded. So our base class is a GriddedProp.
3. We have a number of data files where the default names could be 'water_t' or 'temp', and we want to write them in to be auto-detected.

Here is the class definition: ::

    class WaterTemperature(GriddedProp):
        default_names = ['water_t', 'temp']

That's it! Now, you can do the following in your scripts: ::

    from gnome.environment.property_classes import WaterTemperature

    point = [50.3, 40.2]
    fn = 'my_datafile.nc'
    temp = WaterTemperature.from_netCDF(filename=fn)
    first_temp_at_point = temp.at(point, temp.time.min_time)

Lets do a more advanced example. We want to do another WaterTemperature object, but we want to
force only a single depth, even if our data has multiple layers. If a user asks for the temperature
at a point anywhere in the water column, it will only give the value at the surface (or bottom). This
is not very useful except as a demonstration. ::

    class WaterTemperature(GriddedProp):
        default_names = ['water_t', 'temp']

        def at(self, points, time, units=None, depth=-1, extrapolate = False):
            return super(GriddedProp, self).at(points, time, units=units, depth=0, extrapolate=extrapolate)

The point is if a property subclass needs to implement some special logic, it can be accomplished by simply overriding the function and writing it in.