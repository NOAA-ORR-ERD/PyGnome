#!/usr/bin/env python

"""
outputters.py

module to define classes for GNOME output:
  - image writting
  - saving to netcdf
  - saving to other formats ?

"""

class Outputter(object):
    """
    base class fr all outputters
    """

    def prepare_for_model_run:
        """
        Do what you need to do to prepare for a new mode run
        """
        pass

    def prepare_for_model_step:
        """
        Do what you need to do to prepare for a new model step
        """
        pass
        