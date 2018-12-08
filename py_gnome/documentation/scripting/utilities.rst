Scripting utilities
====================================================

GNOME has a scripting module where many utilties and helper functions can be accessed to simplify common tasks.
These include helper functions for easier creation of certain types of spills and movers.

To make it easier to write simple scripts, we have set it up so that you can import all teh names in the scripting module with::


    from gnome.scripting import *

If you want to keep your scripts namespace clean, you can also import them with a short name::

    import gnome.scripting as gs

You can now access the names in the scipting module with, e.g. ``gs.Model``

Examples in this document will use the ``import *`` approach


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

.. automodule:: gnome.scripting
    :members: constant_wind, constant_wind_mover, wind_mover_from_file, make_images_dir,
              surface_point_line_spill, subsurface_plume_spill, hours






