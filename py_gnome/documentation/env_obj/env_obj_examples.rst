Examples
========

Create GridCurrent from a netCDF file.::

    import numpy as np
    import netCDF4 as nc4
    from datetime import datetime, timedelta
    from gnome.environment import GridCurrent

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

One major advantage to environment objects is re-use of common attributes. For example, in a data file you have a grid, and
wind and current variables that are associated with the grid. ::

    current = GridCurrent.from_netCDF(filename = fn)
    wind = GridWind.from_netCDF(filename = fn,
                                grid = current.grid)

In the above example, the current and wind now both share the same grid object, which has numerous performance benefits. This is
one of the most common cases of sharing between Environment objects.

You can create an environment out of an already open dataset as well. This may help alleviate 'too many files' problems when working
with large numbers of files::

    df = netCDF4.Dataset(fn)
    current = GridCurrent.from_netCDF(dataset=df)

You can also set up an environment object from scratch, which can be very useful when mocking up a situation. The following code creates
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

    #Create environment objects for the components separately, then combine into a GridCurrent
    vels_x = GriddedProp(name='v_x',units='m/s',time=[t], grid=g, data=vx)
    vels_y = GriddedProp(name='v_y',units='m/s',time=[t], grid=g, data=vy)
    vg = GridCurrent(variables = [vels_y, vels_x], time=[t], grid=g, units='m/s')

Defining a new environment object
------------------------

To create a new environment object, let us take the example of water temperature.

1. It is scalar, so it would inherit from a scalar-type environment object
2. In this example, it is gridded. So our base class is a GriddedProp.
3. We have a number of data files where the default names could be 'water_t' or 'temp', and we want to write them in to be auto-detected.

Here is the class definition: ::

    class WaterTemperature(GriddedProp):
        default_names = ['water_t', 'temp']

That's it! Now, you can do the following in your scripts: ::

    from gnome.environment.environment_objects import WaterTemperature

    point = [50.3, 40.2]
    fn = 'my_datafile.nc'
    temp = WaterTemperature.from_netCDF(filename=fn)
    first_temp_at_point = temp.at(point, temp.time.min_time)

Lets do a more advanced example. 




