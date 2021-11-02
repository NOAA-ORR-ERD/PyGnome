###############
Testing Scripts
###############

This directory contains various complete scripts for running GNOME

Each was designed to exercise a particular py_gnome feature.

They can be used both to test the ``gnome`` package and to see how some of the more complicated features can be used.

For example of practical scripts, see the ``example_scripts`` directory.

Script Structure
================

These scripts are structured to be run in a batch, so may not reflect how to write a simple script for your own purposes.

They also have a few features to make them suitable for testing and using in a git repo, for example, auto-downloading data files.

Each script can be run by itself, e.g. ::


  cd script_long_island
  python script_long_island.py

Or they can be run in a batch mode::

  python run_all.py

That script will output a report of which scripts ran successfully, and which resulted in errors.

There is also a ``script_runner.py`` that can do more fine-grained running,
but it's not all that useful anymore ;-)




