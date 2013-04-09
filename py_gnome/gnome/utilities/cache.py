#!/usr/bin/env python

"""
cache system for caching element data on disk for
accessing again for output, etc.

"""
import os
import warnings
import tempfile
import shutil
import copy

import numpy as np

from gnome.spill_container import SpillContainerData, SpillContainerPairData

# create a temp dir for this python instance
# this should happen once, on first import
# it will get cleaned up when python exits
# all individual cache dirs go in this one.
_cache_dir = tempfile.mkdtemp()


class CacheError(Exception):
    """
    Here so we can be sure the user knows the error is coming from here
    """
    pass

def clean_up_cache(dir_name=_cache_dir):
    """
    Deletes a cache dir.

    Designed to be called at program exit. and/or when
    individual cache objects are deleted

    raises a warning if there is problem deleting a particular directory
    """
    #delete it:
    try:
        shutil.rmtree(dir_name)
    except OSError: # this would happen if the dir is already deleted
        ## note: should be smarter and check the error code in 
        ##       the Exception to make sure that it's a "file not there"
        pass 
    except Exception as excp: # something else went wrong
        warnings.warn( "Problem Deleting cache dir" )
        warnings.warn( repr(excp) ) # using repr to get the Error type in the warning

# need to clean up temp directories at exit:
# this will clean up the master temp dir, and anything in it if
# something went wrong with __del__ in the individual objects
import atexit
atexit.register(clean_up_cache)


class ElementCache(object):
    """
    cache for element data -- i.e. the data associated with the particles

    this caches UncertainSpillContainerPair

    The cache can be accessed to re-draw the LE movies, etc.

    """
    def __init__(self, cache_dir=None, enabled=True):
        """
        initialize a new cache object

        :param cache_dir=None: full path to the dir the cache should be stored in.
                               if not provided, a temp dir will be created by the
                               python tempfile module
        """
   
        if cache_dir is None:
            self._cache_dir = os.path.join( tempfile.mkdtemp(dir=_cache_dir) )
        else:
            self._cache_dir = cache_dir

        # dict to hold recent data so we don't need to pull from the filesystem
        self.recent = {}
        self.enabled = True # flag for whther to enable disk cache

    def __del__(self):
        """
        clear out the cache when this object is deleted
        """
        clean_up_cache(dir_name=self._cache_dir)

    def _make_filename(self, step_num, uncertain=False):
        """
        returns a filename of the temp file generated from step_num

        :param step_num: the model step number that is saved/reloaded

        This here so that loading and saving use the same code
        """
        if uncertain:
            return os.path.join(self._cache_dir, "step_%06i_uncert.npz"%step_num)
        else:
            return os.path.join(self._cache_dir, "step_%06i.npz"%step_num)

    def save_timestep(self, step_num, spill_container_pair):
        """
        add a time step of data to the cache

        :param step_num: the step number of the data
        :param spill_container: the spill container at this step

        """
        for sc in spill_container_pair.items():
            data = copy.deepcopy(sc.data_arrays_dict)
            # save a copy of the most recent in memory:
            #   this could be made smarter, to hold more later
            ## note: this assume that the certain SC will be first!
            if sc.uncertain:
                self.recent[step_num][1] = data
            else:
                #this creates a new dict, so only one step is saved
                self.recent = { step_num: [data, None] }
            # write the data if enabled
            # could be threaded -- data is a copy, so doesn't need ot be re-used by anything
            if self.enabled:
                if sc.uncertain:
                    np.savez(self._make_filename(step_num, True), **data)
                else:
                    np.savez(self._make_filename(step_num), **data)

    def load_timestep(self, step_num):
        """
        Returns a SpillContainer with the data arrays cached on disk

        :param step_num: the step number you want to load.

        """

        # look first in in-memory cache.
        try:
            (data_arrays_dict, u_data_arrays_dict) = self.recent[step_num]
        except KeyError:
            # not in the recent dict: try to load from disk
            try:
                data_arrays_dict = np.load(self._make_filename(step_num))
            except IOError:
                raise CacheError("step: %i is not in the cache"%step_num)
            try: # look for an uncertain one:
                u_data_arrays_dict = np.load(self._make_filename(step_num, True))
            except IOError:
                u_data_arrays_dict = None

        #Build a SpillContainerPair:
        sc = SpillContainerData(data_arrays_dict)
        if u_data_arrays_dict is None:
            u_sc = None
        else:
            u_sc = SpillContainerData(u_data_arrays_dict, uncertain=True)
        scp = SpillContainerPairData(sc, u_sc)

        return scp

    def rewind(self):
        """
        rewinds the cache -- clearing out everything
        """
        
        # clean out the in-memory cache
        self.recent = {}
        
        # clean out the disk cache
        shutil.rmtree(self._cache_dir)
        os.mkdir(self._cache_dir)
        


