#!/usr/bin/env python

"""
unit tests for the cache system

designed to be run with py.test
"""

import os

import numpy as np

from datetime import datetime, timedelta

import pytest

from gnome.utilities import cache

from gnome.spill_container import SpillContainerPairData

from ..conftest import sample_sc_release

# some sample datetimes for tests:

dt = datetime(2013, 4, 15, 12)
tdelta = timedelta(hours=1)


def test_init():
    """
    can we even create one?
    """
    c = cache.ElementCache()
    assert True

@pytest.mark.skip("these are intermittently failing -- and we're not using the cache anyway")
def test_cache_clear_on_delete():

    c1 = cache.ElementCache()
    d1 = c1._cache_dir
    c2 = cache.ElementCache()
    d2 = c2._cache_dir
    c3 = cache.ElementCache()
    d3 = c3._cache_dir
    c4 = cache.ElementCache()
    d4 = c4._cache_dir
    assert os.path.isdir(d1)
    assert os.path.isdir(d2)
    assert os.path.isdir(d3)
    assert os.path.isdir(d4)

    del c1
    assert not os.path.isdir(d1)

    del c2
    assert not os.path.isdir(d2)

#    del c3
#    assert not os.path.isdir(d3)

#    del c4
#    assert not os.path.isdir(d4)


def test_write():

    # create a spill_container to save:

    sc = sample_sc_release(num_elements=10, start_pos=(3.14, 2.72, 1.2))

    # add a timestamp:

    sc.current_time_stamp = dt

    # put it in a SpillContainerPair

    scp = SpillContainerPairData(sc)

    # create a cache object:

    c = cache.ElementCache()

    # save it:

    c.save_timestep(0, scp)


def test_write_uncert():

    # create a spill_container to save:

    sc = sample_sc_release(num_elements=10, start_pos=(3.14, 2.72, 1.2))
    u_sc = sample_sc_release(num_elements=10, start_pos=(4.14, 3.72, 2.2),
                             uncertain=True)

    # add a timestamp:

    sc.current_time_stamp = dt

    # put it in a SpillContainerPair

    scp = SpillContainerPairData(sc, u_sc)

    # create a cache object:

    c = cache.ElementCache()

    # save it:

    c.save_timestep(0, scp)


def test_write_and_read_back():
    """
    write to cahce an read back

    no uncertainty
    """

    # create a cache object:

    c = cache.ElementCache()

    # create a spill_container to save:

    sc = sample_sc_release(num_elements=10, start_pos=(3.14, 2.72, 1.2))

    # add a timestamp:

    sc.current_time_stamp = dt

    # put it in a SpillContainerPair

    scp = SpillContainerPairData(sc)

    # make a copy of positons for later testing

    pos0 = sc['positions'].copy()
    c.save_timestep(0, scp)

    # change things...

    sc['positions'] += 1.1
    pos1 = sc['positions'].copy()

    # change time stamp

    sc.current_time_stamp = dt + tdelta
    c.save_timestep(1, scp)

    # change things...

    sc['positions'] *= 1.1
    pos2 = sc['positions'].copy()

    # change time stamp

    sc.current_time_stamp = dt + tdelta * 2

    # save it:

    c.save_timestep(2, scp)

    # read them back

    sc2 = c.load_timestep(2)
    assert np.array_equal(sc2._spill_container['positions'], pos2)
    assert sc2._spill_container.current_time_stamp == dt + tdelta * 2

    sc0 = c.load_timestep(0)
    assert np.array_equal(sc0._spill_container['positions'], pos0)
    assert sc0._spill_container.current_time_stamp == dt

    sc1 = c.load_timestep(1)
    assert np.array_equal(sc1._spill_container['positions'], pos1)
    assert sc1._spill_container.current_time_stamp == dt + tdelta

    sc2 = c.load_timestep(2)
    assert np.array_equal(sc2._spill_container['positions'], pos2)
    assert sc2._spill_container.current_time_stamp == dt + tdelta * 2


def test_write_and_read_back_uncertain():
    """
    write to cache and read back

    with uncertainty
    """

    # create a cache object:

    c = cache.ElementCache()

    # create a spill_container to save:

    sc = sample_sc_release(num_elements=10, start_pos=(3.14, 2.72, 1.2))
    u_sc = sample_sc_release(num_elements=10, start_pos=(4.14, 3.72, 2.2),
                             uncertain=True)

    # put it in a SpillContainerPair

    scp = SpillContainerPairData(sc, u_sc)

    # make a copy of positons for later testing

    pos0 = sc['positions'].copy()
    u_pos0 = u_sc['positions'].copy()
    c.save_timestep(0, scp)

    # change things...

    sc['positions'] += 1.1
    u_sc['positions'] += 1.1
    pos1 = sc['positions'].copy()
    u_pos1 = u_sc['positions'].copy()
    c.save_timestep(1, scp)

    # change things...

    sc['positions'] *= 1.1
    pos2 = sc['positions'].copy()

    # save it:

    c.save_timestep(2, scp)

    # read them back

    sc2 = c.load_timestep(2)
    assert np.array_equal(sc2._spill_container['positions'], pos2)
    sc0 = c.load_timestep(0)
    assert np.array_equal(sc0._spill_container['positions'], pos0)
    assert np.array_equal(sc0._u_spill_container['positions'], u_pos0)
    sc1 = c.load_timestep(1)
    assert np.array_equal(sc1._spill_container['positions'], pos1)
    assert np.array_equal(sc1._u_spill_container['positions'], u_pos1)
    sc2 = c.load_timestep(2)
    assert np.array_equal(sc2._spill_container['positions'], pos2)


def test_read_back_from_memory():
    """
    test reading back the last item from the memory cache
    """

    # create a cache object:

    c = cache.ElementCache()

    # create a spill_container to save:

    sc = sample_sc_release(num_elements=10, start_pos=(3.14, 2.72, 1.2))

    # put it in a SpillContainerPair

    scp = SpillContainerPairData(sc)

    c.save_timestep(0, scp)

    # change things...

    sc['positions'] += 1.1
    c.save_timestep(1, scp)

    # clear the cache files (private API...)

    cache.clean_up_cache(dir_name=c._cache_dir)

    # with cache cleared, this shouldn't load

    with pytest.raises(cache.CacheError):
        c.load_timestep(0)

    # but this should

    scp1 = c.load_timestep(1)

    print(scp1)
    print(scp1._spill_container._data_arrays)
    assert np.array_equal(scp1._spill_container['positions'],
                          sc['positions'])


def test_cache_error():
    """
    you should get an exception when you ask for somethign not there
    """

    # create a cache object:

    c = cache.ElementCache()
    with pytest.raises(cache.CacheError):
        c.load_timestep(3)


def test_rewind():
    """
    test that the cache is cleared out after a rewind call
    """

    # create a cache object:

    c = cache.ElementCache()

    # create a set of spill_container to save:

    sc = sample_sc_release(num_elements=10, start_pos=(3.14, 2.72, 1.2))
    u_sc = sample_sc_release(num_elements=10, start_pos=(4.14, 3.72, 2.2),
                             uncertain=True)

    # put it in a SpillContainerPair

    scp = SpillContainerPairData(sc, u_sc)

    # save it

    c.save_timestep(0, scp)

    # change things and save again

    sc['positions'] += 1.1
    u_sc['positions'] += 1.1
    pos1 = sc['positions'].copy()
    u_pos1 = u_sc['positions'].copy()
    c.save_timestep(1, scp)

    # change things and save again

    sc['positions'] *= 1.1
    pos2 = sc['positions'].copy()

    # save it:

    c.save_timestep(2, scp)

    # read them back, just to make sure

    sc2 = c.load_timestep(2)
    sc0 = c.load_timestep(0)
    sc1 = c.load_timestep(1)
    sc2 = c.load_timestep(2)

    # rewind

    c.rewind()

    # make sure nothing is there:

    with pytest.raises(cache.CacheError):
        c.load_timestep(0)
    with pytest.raises(cache.CacheError):
        c.load_timestep(1)
    with pytest.raises(cache.CacheError):
        c.load_timestep(2)

    # make sure it works again:

    c.save_timestep(0, scp)


#    assert False

if __name__ == '__main__':
    test_write_and_read_back()
