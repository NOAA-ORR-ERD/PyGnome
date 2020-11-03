Scripting utilities
===================

The GNOME scripting module contains most of the gnome objects you will need for scripting, and a number of utilities and helper functions that can be accessed to simplify common tasks.


To make it easier to write simple scripts, we have set it up so that you can import all the names in the scripting module with::


    import gnome.scripting as gs

You can now access the names in the scripting module with, e.g. ``gs.Model``

Examples in this document will use the ``import as gs`` approach


.. note::

    The scripting utilities are under active development as we determine which helper functions will be useful to make setting up and running pyGNOME easier.


:mod:`gnome.scripting.time_utils`
---------------------------------

Internally, py_Gnome uses ``datetime.timedelta`` objects to represent time spans. But it is a bit awkward to create these objects::

    datetime.timedelta(seconds=3600)

The time_utils module provides handy utilities to make it easier to construct these objects:

.. automodule:: gnome.scripting.time_utils



:mod:`gnome.scripting`
----------------------

.. autoclass:: gnome.scripting.Model
    :members: __init__

Utilities
---------

.. automodule:: gnome.scripting
    :members: make_images_dir, PrintFinder, get_datafile, remove_netcdf,  set_verbose,


Time Utilities
--------------

.. automodule:: gnome.scripting
    :members: seconds, minutes, hours, days, weeks, InfTime, MinusInfTime, asdatetime


Environment Objects
-------------------

.. automodule:: gnome.scripting
    :members: constant_wind, GridCurrent, IceAwareCurrent, IceAwareWind, Tide,
              Water, Waves, Wind


Movers
------

.. automodule:: gnome.scripting
    :members: constant_wind_mover, CatsMover, ComponentMover, IceAwareRandomMover,
              PyCurrentMover, PyWindMover, RandomMover, RandomMover3D, RiseVelocityMover,
              SimpleMover, WindMover, wind_mover_from_file


Maps
----

.. automodule:: gnome.scripting
    :members: GnomeMap, MapFromBNA


Spills
------

.. automodule:: gnome.scripting
    :members: point_line_release_spill, surface_point_line_spill,
              subsurface_plume_spill, grid_spill, spatial_release_spill,
              subsurface_plume_spill,


Outputters
----------

.. automodule:: gnome.scripting
    :members: Renderer, KMZOutput, NetCDFOutput, OilBudgetOutput, ShapeOutput,
              WeatheringOutput





