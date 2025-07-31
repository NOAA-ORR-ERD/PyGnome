Example data files
==================

This dir has a set of example files that can be used as input to PyGNOME:

PyGNOME needs "drivers" for environmental conditions:
ocean currents, winds, etc. It supports a wide variety of grid types, etc.

But these files need to conform to various standards, use common variable names etc.

In this dir are files known to work with PyGNOME. These can be used both to test that
nothing has broken with files that used to work, and as examples.

Each file has a simple test script attached to it to show it can be used (and that it works)

Accessing the files
-------------------

We don't want to store large binary files in the PyGNOME source repo.

So the actual files are available from the GNOME server at:

https://gnome.orr.noaa.gov/py_gnome_testdata/gridded_test_files/

But each individual file should be auto-downloaded when you run its test script.

You can also download them all by running the ``run_all.py script`` -- it will run all the examples, downloading the files as it goes.

Adding new files
----------------

To add a new example file:

1) Add the file itself (unless its tiny) to the py_gnme_data project.

   https://gitlab.orr.noaa.gov/gnome/data

   Put it in the "gridded_test_files" directory.

   Once you add it, commit it, and push, it will get automatically uploaded to the py_gnome data server where it can be accessed at run time.

2) Create a directory for your example: give as meaningful name as you can.

3) Add a BNA map file (if desired) to the dir (it can be directly added if it's not too big, and try not to make it too big)

4) Add a script that loads and uses your example data file. Call it:

   `something_example_file.py` "something" should be informative, if possible.

   You can optionally use the template in: template_example_file.py

   In the scripts docstring, put whatever helpful notes you can think of:

   where it came from, why it's unique, whatever ...

5) Run your script and make sure it works :-)

6) make sure NOT to add the actual file to git -- these are big, and we don't want to clutter up the repo any more than it is.





