.. _scripts_tutorial:

Sample Scripts
=====================

A good way to learn to use PyGNOME is to find an example script that uses similar coponets what you need, and work on that. the Sample scripts can be found in the ``py_gnome/scripts`` dir, in directories names with a ``script_*``.

There are also a few utilities in the ``scripts`` directory:

``script_runner.py`` will run a particular script, with various options for output: image output, netcdf output, saving to a "save file", or reloading from a "save file". try::

    $ python script_runner -h for help

``run_all.py`` will run all the scripts in the directory -- good for testing.


Below are summaries of a few of them:

``script_marianas``
--------------------

This script used current from the Navy's HYCOM model, subsetted to a region around the Marianas Islands. The land-water map is a BNA pulled from the GOODS shoreline extractor, with a map_bounds hand added to a larger region.

``script_boston``
------------------


``script_long_island``
-----------------------


``script_chesapeake_bay``
--------------------------
    

``script_mariana``
-------------------           

``script_guam``
----------------

``script_mississippi_river``
----------------------------






