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

import numpy
np = numpy

from gnome.spill_container import (SpillContainerData,
                                   SpillContainerPairData)

# create a temp dir for this python instance
# this should happen once, on first import
# it will get cleaned up when python exits
# all individual cache dirs go in this one.

_cache_dir = tempfile.mkdtemp()


class CacheError(Exception):
    'Here so we can be sure the user knows the error is coming from here'
    pass


def clean_up_cache(dir_name=_cache_dir):
    """
    Deletes a cache dir.

    Designed to be called at program exit. and/or when
    individual cache objects are deleted

    raises a warning if there is problem deleting a particular directory
    """
    # delete it:

    try:
        shutil.rmtree(dir_name)
    except OSError:
        # This would happen if the dir is already deleted
        # note: should be smarter and check the error code in
        #       the Exception to make sure that it's a "file not there"
        pass
    except Exception, excp:
        # something else went wrong
        warnings.warn('Problem Deleting cache dir')
        # using repr to get the Error type in the warning
        warnings.warn(repr(excp))


# need to clean up temp directories at exit:
# this will clean up the master temp dir, and anything in it if
# something went wrong with __del__ in the individual objects

import atexit
atexit.register(clean_up_cache)


class ElementCache(object):
    """
    Cache for element data -- i.e. the data associated with the particles.
    This caches UncertainSpillContainerPair
    The cache can be accessed to re-draw the LE movies, etc.

    TODO: This is a really fragile module in terms of handling multiple
          instances.  The __del__() method of previous instances can clear
          the _cache_dir at the whim of the GC.
          We may want to manage this differently.
    """
    def __init__(self, cache_dir=None, enabled=True):
        """
        initialize a new cache object

        :param cache_dir=None: full path to the directory where the cache
                               should be stored.
                               If not provided, a temp dir will be created by
                               the python tempfile module
        """
        if cache_dir is None:
            self._cache_dir = os.path.join(tempfile.mkdtemp(dir=_cache_dir))
        else:
            self._cache_dir = cache_dir

        # dict to hold recent data so we don't need to pull from the
        # file system
        self.recent = {}

        # flag for whether to enable disk cache
        self.enabled = enabled

    def __del__(self):
        'Clear out the cache when this object is deleted'
        clean_up_cache(dir_name=self._cache_dir)

    def _make_filename(self, step_num, uncertain=False):
        """
        Returns a filename of the temp file generated from step_num

        :param step_num: the model step number that is saved/reloaded

        This here so that loading and saving use the same code
        """
        if uncertain:
            return os.path.join(self._cache_dir,
                                'step_%06i_uncert.npz' % step_num)
        else:
            return os.path.join(self._cache_dir,
                                'step_%06i.npz' % step_num)

    def save_timestep(self, step_num, spill_container_pair):
        """
        add a time step of data to the cache

        :param step_num: the step number of the data
        :param spill_container: the spill container at this step
        """
        for sc in spill_container_pair.items():
            data = copy.deepcopy(sc.data_arrays)

            if sc.current_time_stamp:
                data['current_time_stamp'] = np.array(sc.current_time_stamp)

            # # note: this assumes that the certain SC will be first!

            if sc.uncertain:
                self.recent[step_num][1] = data
            else:
                # this creates a new dict, so only one step is saved
                self.recent = {step_num: [data, None]}

            # write the data if enabled
            # could be threaded -- data is a copy, so doesn't need to be
            #                      re-used by anything
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
            # make a copy because we pop out the current_time_stamp
            # make these changes to the copy so the self.recent does not change

            (data_arrays, u_data_arrays) = copy.deepcopy(self.recent[step_num])

            # copy.deepcopy(self.recent[step_num]) converts
            # 'current_time_stamp' to datetime object
            # To be consistent with np.load() operation below,
            # make this an array.
            if 'current_time_stamp' in data_arrays:
                data_arrays['current_time_stamp'] = \
                    np.array(data_arrays['current_time_stamp'])
                if u_data_arrays:
                    u_data_arrays['current_time_stamp'] = \
                        np.array(u_data_arrays['current_time_stamp'])
        except KeyError:
            # not in the recent dict: try to load from disk
            try:
                data_arrays = \
                    dict(np.load(self._make_filename(step_num)))
            except IOError:
                raise CacheError('step: {0} is not in the cache'
                                 .format(step_num))

            try:
                u_data_arrays = \
                    dict(np.load(self._make_filename(step_num, True)))
            except IOError:
                u_data_arrays = None

        # HOWEVER, loading numpy arrays
        #     data_arrays = dict(np.load(self._make_filename(step_num)))
        # converts current_time_stamp to numpy.ndarray objects
        current_time_stamp = None
        if 'current_time_stamp' in data_arrays:
            current_time_stamp = data_arrays.pop('current_time_stamp').item()

        sc = SpillContainerData(data_arrays)
        if current_time_stamp:
            sc.current_time_stamp = current_time_stamp

        if u_data_arrays is None:
            u_sc = None
        else:
            current_time_stamp = None
            if 'current_time_stamp' in u_data_arrays:
                current_time_stamp = \
                    u_data_arrays.pop('current_time_stamp').item()

            u_sc = SpillContainerData(u_data_arrays, uncertain=True)

            if current_time_stamp:
                u_sc.current_time_stamp = current_time_stamp

        scp = SpillContainerPairData(sc, u_sc)

        return scp

    def rewind(self):
        'Rewinds the cache -- clearing out everything'
        # clean out the in-memory cache
        self.recent = {}

        # clean out the disk cache
        if os.path.isdir(self._cache_dir):
            shutil.rmtree(self._cache_dir)
        os.mkdir(self._cache_dir)
