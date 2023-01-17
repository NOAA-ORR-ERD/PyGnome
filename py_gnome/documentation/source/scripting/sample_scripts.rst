.. _sample_scripts:

Sample Scripts Included with PyGNOME
####################################


A good way to learn to use PyGNOME is to find an example script that uses similar components to what you need, and use that as a starting point.

There are two sets of scripts included with PyGNOME:

* ``py_gnome/scripts/example_scripts``: simple examples for using PyGNOME for common use cases.


* ``py_gnome/scripts/testing_scripts``: scripts set up to test particular aspects of PyGNOME


Testing Scripts
===============

The testing scripts can be found in the ``py_gnome/scripts/testing_scripts`` dir, in directories named ``script_*``. Each of these contains a ``make_model()`` function that builds up a PYGNOME Model configuration, and then the model is run in the ``if __name__ == "__main"`` stanza. This is to facilitate them being automatically run -- it is not necessary to separate these steps in your own scripts.

There are also a few utilities in the ``testing_scripts`` directory:

``script_runner.py`` will run a particular script, with various options for output: image output, netcdf output, saving to a "save file", or reloading from a "save file". try::

    $ python script_runner -h for help

``run_all.py`` will run all the scripts in the directory -- good for testing.

But for the most part, you will want to run each script by itself, e.g.::

    $ python script_marianas.py


Below are summaries of a few of them:


``script_marianas``
-------------------

This script used current from the Navy's HYCOM model, subsetted to a region around the Marianas Islands. 
The land-water map is a BNA pulled from the GOODS shoreline extractor, with a map_bounds hand added to a larger region.

``script_boston``
------------------
This script models the Boston & Vicinity location file which includes a component mover (wind-driven currents).


``script_long_island``
-----------------------
This script models the Long Island location file - a single CATS pattern, wind, and diffusion.


``script_chesapeake_bay``
--------------------------
This script uses a gridded time dependent current pattern for Chesapeake Bay.   


``script_passamaquoddy``
------------------------
This script uses the current cycle mover for Passamaquoddy Bay - a set of representative patterns driven by a tide.
    

``script_mariana``
------------------

``script_guam``
----------------

``script_mississippi_river``
----------------------------

``script_sf_wind``
------------------
This script uses a gridded wind.


``script_weatherers``
---------------------
This script uses all weatherers and response options (as of 9/2015).
Weatherers - evaporation, natural dispersion, sedimentation, and emulsification.
Response options - skimmers, chemical dispersion, and burning.



