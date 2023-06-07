#!/usr/bin/env python
'''
Tests for netcdf_outputter
'''

import os
from datetime import datetime, timedelta
from math import ceil

import pytest
from pytest import raises

import numpy as np

import netCDF4 as nc

from gnome.spills import surface_point_line_spill, Spill, Release
from gnome.spill_container import SpillContainerPair
from gnome.weatherers import Evaporation
from gnome.environment import Water
from gnome.movers import RandomMover, constant_point_wind_mover
from gnome.outputters import NetCDFOutput
from gnome.model import Model
from ..conftest import test_oil

# file extension to use for test output files
#  this is used by the output_filename fixture in conftest:
FILE_EXTENSION = ".nc"


@pytest.fixture(scope='function')
def model(sample_model_fcn, output_filename):
    """
    Use fixture model_surface_release_spill and add a few things to it for the
    test
    """
    model = sample_model_fcn['model']

    model.cache_enabled = True
    model.spills += \
        surface_point_line_spill(num_elements=5,
                                 start_position=sample_model_fcn['release_start_pos'],
                                 release_time=model.start_time,
                                 end_release_time=model.start_time + model.duration,
                                 substance=test_oil,
                                 amount=1000,
                                 units='kg')

    water = Water()
    model.movers += RandomMover(diffusion_coef=100000)
    model.movers += constant_point_wind_mover(1.0, 0.0)
    model.weatherers += Evaporation(water=water, wind=model.movers[-1].wind)

    model.outputters += NetCDFOutput(output_filename)

    model.rewind()

    return model


def test_init_exceptions():
    '''
    test exceptions raised during __init__
    '''
    with raises(ValueError):
        # must be filename, not dir name
        NetCDFOutput(os.path.abspath(os.path.dirname(__file__)))

    with raises(ValueError):
        NetCDFOutput('invalid_path_to_file/file.nc')


def test_exceptions(output_filename):
    spill_pair = SpillContainerPair()

    print("output_filename:", output_filename)
    # begin tests
    netcdf = NetCDFOutput(output_filename, which_data='all')
    netcdf.rewind()  # delete temporary files

    with raises(TypeError):
        # need to pass in model start time
        netcdf.prepare_for_model_run(num_time_steps=4)

    with raises(TypeError):
        # need to pass in model start time and spills
        netcdf.prepare_for_model_run()

    with raises(ValueError):
        # need a cache object
        netcdf.write_output(0)

    with raises(ValueError):
        netcdf.which_data = 'some random string'

    # changed renderer and netcdf ouputter to delete old files in
    # prepare_for_model_run() rather than rewind()
    # -- rewind() was getting called a lot
    # -- before there was time to change the ouput file names, etc.
    # So for this unit test, there should be no exception if we do it twice.
    netcdf.prepare_for_model_run(model_start_time=datetime.now(),
                                 spills=spill_pair,
                                 num_time_steps=4)
    netcdf.prepare_for_model_run(model_start_time=datetime.now(),
                                 spills=spill_pair,
                                 num_time_steps=4)

    with raises(AttributeError):
        'cannot change after prepare_for_model_run has been called'
        netcdf.prepare_for_model_run(model_start_time=datetime.now(),
                                     spills=spill_pair,
                                     num_time_steps=4)
        netcdf.which_data = 'most'


def test_exceptions_middle_of_run(model):
    """
    Test attribute exceptions are called when changing parameters in middle of
    run for 'which_data' and 'filename'
    """
    model.rewind()
    model.step()
    o_put = [model.outputters[outputter.id]
             for outputter in model.outputters
             if isinstance(outputter, NetCDFOutput)][0]

    assert o_put.middle_of_run

    with raises(AttributeError):
        o_put.filename = 'test.nc'

    with raises(AttributeError):
        o_put.which_data = True

    with raises(AttributeError):
        o_put.compress = False

    with raises(AttributeError):
        o_put.chunksize = 1024 * 1024

    model.rewind()

    assert not o_put.middle_of_run

    o_put.compress = True


def test_prepare_for_model_run(model):
    """
    Use model fixture.
    Call prepare_for_model_run for netcdf_outputter

    Simply asserts the correct files are created and no errors are raised.
    """
    for outputter in model.outputters:
        if isinstance(outputter, NetCDFOutput):  # there should only be 1!
            o_put = model.outputters[outputter.id]

    model.rewind()
    model.step()  # should call prepare_for_model_step

    assert os.path.exists(o_put.filename)

    if model.uncertain:
        assert os.path.exists(o_put._u_filename)
    else:
        assert not os.path.exists(o_put._u_filename)

    print(o_put.filename)

def test_variable_attributes(model):
    """
    Call prepare_for_model_run for netcdf_outputter

    The netcdf file should have been created with core variables

    This checks the status codes only for now.
    """
    for outputter in model.outputters:
        if isinstance(outputter, NetCDFOutput):  # there should only be 1!
            o_put = model.outputters[outputter.id]

    model.rewind()
    model.step()  # should call prepare_for_model_step

    # just to get an error early!
    assert os.path.exists(o_put.filename)

    ds = nc.Dataset(o_put.filename)

    # print(ds)

    sc = ds.variables['status_codes']
    assert sc.long_name == 'particle status code'
    for val in sc.flag_values:
        val = int(val)  # just making sure it's an integer
    # print(sc.flag_values) #: [v.value for v in oil_status],
    print(sc.flag_meanings) # : " ".join("{}:{}".format(v.name, v.value)
                            #       for v in oil_status)
    # parse flag meanings to make sure it's the right format
    #flags = sc.flag_meanings.split()
    for flag in sc.flag_meanings.split():
        val, name = flag.split(':')
        val = int(val)
        print(val, name)


# @pytest.mark.slow
def test_write_output_standard(model):
    """
    Rewind model defined by model fixture.

    invoke model.step() till model runs all 5 steps

    For each step, compare the standard variables in the model.cache to the
    data read back in from netcdf files. Compare uncertain and uncertain data.

    Since 'latitude', 'longitude' and 'depth' are float 32 while the data in
    cache is float64, use np.allclose to check it is within 1e-5 tolerance.
    """
    model.rewind()
    _run_model(model)

    # check contents of netcdf File at multiple time steps
    # (there should only be 1!)
    o_put = [model.outputters[outputter.id]
             for outputter in model.outputters
             if isinstance(outputter, NetCDFOutput)][0]

    atol = 1e-5
    rtol = 0

    uncertain = False
    for file_ in (o_put.filename, o_put._u_filename):
        with nc.Dataset(file_) as data:
            dv = data.variables
            time_ = nc.num2date(dv['time'][:], dv['time'].units,
                                calendar=dv['time'].calendar)

            idx = np.cumsum((dv['particle_count'])[:])
            idx = np.insert(idx, 0, 0)  # add starting index of 0

            for step in range(model.num_time_steps):
                scp = model._cache.load_timestep(step)

                # check time
                # conversion from floats to datetime can be off by microseconds
                # fixme: this should probably round!
                #        this may help:
                #        https://stackoverflow.com/questions/3463930/how-to-round-the-minute-of-a-datetime-object-python
                print("***** scp timestamp", scp.LE('current_time_stamp',
                                                    uncertain))
                print("***** netcdf time:", time_[step])
                print(type(time_[step]))
                assert scp.LE('current_time_stamp', uncertain) == time_[step].replace(microsecond=0)

                assert np.allclose(scp.LE('positions', uncertain)[:, 0],
                                   (dv['longitude'])[idx[step]:idx[step + 1]],
                                   rtol, atol)
                assert np.allclose(scp.LE('positions', uncertain)[:, 1],
                                   (dv['latitude'])[idx[step]:idx[step + 1]],
                                   rtol, atol)
                assert np.allclose(scp.LE('positions', uncertain)[:, 2],
                                   (dv['depth'])[idx[step]:idx[step + 1]],
                                   rtol, atol)

                assert np.all(scp.LE('spill_num', uncertain)[:] ==
                              (dv['spill_num'])[idx[step]:idx[step + 1]])
                assert np.all(scp.LE('id', uncertain)[:] ==
                              (dv['id'])[idx[step]:idx[step + 1]])
                assert np.all(scp.LE('status_codes', uncertain)[:] ==
                              (dv['status_codes'])[idx[step]:idx[step + 1]])

                # flag variable is not currently set or checked

                if 'mass' in scp.LE_data:
                    assert np.all(scp.LE('mass', uncertain)[:] ==
                                  (dv['mass'])[idx[step]:idx[step + 1]])

                if 'age' in scp.LE_data:
                    assert np.all(scp.LE('age', uncertain)[:] ==
                                  (dv['age'])[idx[step]:idx[step + 1]])

            print('data in model matches output in {0}'.format(file_))

        # 2nd time around, we are looking at uncertain filename so toggle
        # uncertain flag

        uncertain = True


@pytest.mark.slow
def test_write_output_all_data(model):
    """
    rewind model defined by model fixture.
    invoke model.step() till model runs all 5 steps

    For each step, compare the non-standard variables in the model.cache to the
    data read back in from netcdf files. Compare uncertain and uncertain data.

    Only compare the remaining data not already checked in
    test_write_output_standard
    """
    # check contents of netcdf File at multiple time steps (there should only
    # be 1!)

    model.rewind()

    o_put = [model.outputters[outputter.id]
             for outputter in model.outputters
             if isinstance(outputter, NetCDFOutput)][0]
    o_put.which_data = 'all'  # write all data

    _run_model(model)

    uncertain = False
    for file_ in (o_put.filename, o_put._u_filename):
        with nc.Dataset(file_) as data:
            idx = np.cumsum((data.variables['particle_count'])[:])
            idx = np.insert(idx, 0, 0)  # add starting index of 0

            for step in range(model.num_time_steps):
                scp = model._cache.load_timestep(step)
                for var_name in o_put.arrays_to_output:
                    # special_case 'positions'
                    if var_name == 'longitude':
                        nc_var = data.variables[var_name]
                        sc_arr = scp.LE('positions', uncertain)[:, 0]
                    elif var_name == 'latitude':
                        nc_var = data.variables[var_name]
                        sc_arr = scp.LE('positions', uncertain)[:, 1]
                    elif var_name == 'depth':
                        nc_var = data.variables[var_name]
                        sc_arr = scp.LE('positions', uncertain)[:, 2]
                    else:
                        nc_var = data.variables[var_name]
                        sc_arr = scp.LE(var_name, uncertain)
                    if var_name == "surface_concentration":
                        continue
                    if len(sc_arr.shape) == 1:
                        assert np.all(nc_var[idx[step]:idx[step + 1]] ==
                                      sc_arr)
                    elif len(sc_arr.shape) == 2:
                        assert np.all(nc_var[idx[step]:idx[step + 1], :] ==
                                      sc_arr)
                    else:
                        raise ValueError("haven't written a test "
                                         "for 3-d arrays")

        # 2nd time around, we are looking at uncertain filename so toggle
        # uncertain flag
        uncertain = True


def test_run_without_spills(model):
    for spill in model.spills:
        del model.spills[spill.id]

    assert len(model.spills) == 0

    _run_model(model)


@pytest.mark.slow
def test_read_data_exception(model):
    """
    tests the exception is raised by read_data when file contains more than one
    output time and read_data is not given the output time to read
    """
    model.rewind()

    # check contents of netcdf File at multiple time steps (should only be 1!)
    o_put = [model.outputters[outputter.id]
             for outputter in model.outputters
             if isinstance(outputter, NetCDFOutput)][0]

    _run_model(model)

    with raises(ValueError):
        NetCDFOutput.read_data(o_put.filename)


#@pytest.mark.slow
def test_single_time(model):
    """
    tests the option to output only a single time step with output_single_step = True

    sets output_zero_step and output_last_step to False.

    read_data will give an error if there is more than one time in the file
    when neither a time nor an index is specified
    """
    model.rewind()

    o_put = [model.outputters[outputter.id] for outputter in
             model.outputters if isinstance(outputter, NetCDFOutput)][0]
    #o_put.output_timestep = timedelta(seconds=0) # previous method set time step to zero
    o_put.output_timestep = timedelta(seconds=900)
    o_put.output_single_step = True
    curr_time = model.start_time + timedelta(seconds=900)
    o_put.output_start_time = curr_time
    o_put.output_zero_step = False
    o_put.output_last_step = False
    _run_model(model)

    file_ = o_put.filename

    (nc_data, weathering_data) = NetCDFOutput.read_data(file_, curr_time)
    assert curr_time == nc_data['current_time_stamp'].item()
    (nc_data, weathering_data) = NetCDFOutput.read_data(file_)
    assert curr_time == nc_data['current_time_stamp'].item()


@pytest.mark.slow
@pytest.mark.parametrize(("output_ts_factor", "use_time"),
                         [(1, True), (1, False),
                          (2.4, True), (2.4, False),
                          (3, True), (3, False)])
def test_read_standard_arrays(model, output_ts_factor, use_time):
    """
    tests the data returned by read_data is correct when `which_data` flag is
    'standard'. It is only reading the standard_arrays

    Test will only verify the data when time_stamp of model matches the
    time_stamp of data written out. output_ts_factor means not all data is
    written out.

    The use_time flag says data is read by timestamp. If false, then it is read
    by step number - either way, the result should be the same
    """
    model.rewind()

    # check contents of netcdf File at multiple time steps (should only be 1!)
    o_put = [model.outputters[outputter.id] for outputter in
             model.outputters if isinstance(outputter, NetCDFOutput)][0]
    o_put.output_timestep = timedelta(seconds=model.time_step *
                                      output_ts_factor)
    _run_model(model)

    atol = 1e-5
    rtol = 0

    uncertain = False
    for file_ in (o_put.filename, o_put._u_filename):
        _found_a_matching_time = False

        for idx, step in enumerate(range(0, model.num_time_steps,
                                   int(ceil(output_ts_factor)))):
            scp = model._cache.load_timestep(step)
            curr_time = scp.LE('current_time_stamp', uncertain)
            if use_time:
                (nc_data, weathering_data) = NetCDFOutput.read_data(file_,
                                                                    curr_time)
            else:
                (nc_data, weathering_data) = NetCDFOutput.read_data(file_,
                                                                    index=idx)

            # check time
            if curr_time == nc_data['current_time_stamp'].item():
                _found_a_matching_time = True

                # check standard variables
                assert np.allclose(scp.LE('positions', uncertain),
                                   nc_data['positions'], rtol, atol)
                assert np.all(scp.LE('spill_num', uncertain)[:] ==
                              nc_data['spill_num'])
                assert np.all(scp.LE('status_codes', uncertain)[:] ==
                              nc_data['status_codes'])

                # flag variable is not currently set or checked

                if 'mass' in scp.LE_data:
                    assert np.all(scp.LE('mass', uncertain)[:] ==
                                  nc_data['mass'])

                if 'age' in scp.LE_data:
                    assert np.all(scp.LE('age', uncertain)[:] ==
                                  nc_data['age'])

                if uncertain:
                    sc = list(scp.items())[1]
                else:
                    sc = list(scp.items())[0]

                assert sc.mass_balance == weathering_data
            else:
                raise Exception('Assertions not tested since no data found '
                                'in NetCDF file for timestamp: {0}'
                                .format(curr_time))

        if _found_a_matching_time:
            print(('\n'
                   'data in model matches for output in\n'
                   '{0}\n'
                   'and output_ts_factor: {1}'
                   .format(file_, output_ts_factor)))

        # 2nd time around, look at uncertain filename so toggle uncertain flag
        uncertain = True


@pytest.mark.slow
def test_read_all_arrays(model):
    """
    tests the data returned by read_data is correct
    when `which_data` flag is 'all'.
    """
    model.rewind()

    o_put = [model.outputters[outputter.id]
             for outputter in model.outputters
             if isinstance(outputter, NetCDFOutput)][0]

    o_put.which_data = 'all'

    _run_model(model)

    atol = 1e-5
    rtol = 0

    uncertain = False
    for file_ in (o_put.filename, o_put._u_filename):
        _found_a_matching_time = False
        for step in range(model.num_time_steps):
            scp = model._cache.load_timestep(step)
            curr_time = scp.LE('current_time_stamp', uncertain)

            (nc_data, mb) = NetCDFOutput.read_data(file_, curr_time,
                                                   which_data='all')

            if curr_time == nc_data['current_time_stamp'].item():
                _found_a_matching_time = True
                for key in scp.LE_data:
                    if key == 'current_time_stamp':
                        """ already matched """
                        continue
                    elif key == 'positions':
                        assert np.allclose(scp.LE('positions', uncertain),
                                           nc_data['positions'], rtol, atol)
                    elif key == 'mass_balance':
                        assert scp.LE(key, uncertain) == mb
                    else:
                        # not always there
                        if key not in ['surface_concentration']:
                            assert np.all(scp.LE(key, uncertain)[:] ==
                                          nc_data[key])

        if _found_a_matching_time:
            print(('\ndata in model matches for output in \n{0}'.format(file_)))

        # 2nd time around, look at uncertain filename so toggle uncertain flag
        uncertain = True


@pytest.mark.slow
@pytest.mark.parametrize("output_ts_factor", [1, 2])
def test_write_output_post_run(model, output_ts_factor):
    """
    Create netcdf file post run from the cache. Under the hood, it is simply
    calling write_output so no need to check the data is correctly written
    test_write_output_standard already checks data is correctly written.

    Instead, make sure if output_timestep is not same as model.time_step,
    then data is output at correct time stamps
    """
    model.rewind()

    o_put = [model.outputters[outputter.id] for outputter in
             model.outputters if isinstance(outputter, NetCDFOutput)][0]
    o_put.which_data = 'standard'
    o_put.output_timestep = timedelta(seconds=model.time_step * output_ts_factor)

    del model.outputters[o_put.id]  # remove from list of outputters

    _run_model(model)

    # clear out old files...
    o_put.clean_output_files()
    assert not os.path.exists(o_put.filename)

    if o_put._u_filename:
        assert (not os.path.exists(o_put._u_filename))

    # now write netcdf output
    o_put.write_output_post_run(model.start_time,
                                model.num_time_steps,
                                spills=model.spills,
                                cache=model._cache,
                                uncertain=model.uncertain)

    assert os.path.exists(o_put.filename)
    if model.uncertain:
        assert os.path.exists(o_put._u_filename)

    uncertain = False
    for file_ in (o_put.filename, o_put._u_filename):
        ix = 0  # index for grabbing record from NetCDF file
        for step in range(0, model.num_time_steps,
                          int(ceil(output_ts_factor))):
            print("step: {0}".format(step))
            scp = model._cache.load_timestep(step)
            curr_time = scp.LE('current_time_stamp', uncertain)

            (nc_data, mb) = NetCDFOutput.read_data(file_, curr_time)
            assert curr_time == nc_data['current_time_stamp'].item()

            # test to make sure data_by_index is consistent with _cached data
            # This is just to double check that getting the data by curr_time
            # does infact give the next consecutive index
            (data_by_index, mb) = NetCDFOutput.read_data(file_, index=ix)
            assert curr_time == data_by_index['current_time_stamp'].item()
            assert scp.LE('mass_balance', uncertain) == mb

            ix += 1

        if o_put.output_last_step and step < model.num_time_steps - 1:
            '''
            Last timestep written to NetCDF wasn't tested - do that here
            '''
            scp = model._cache.load_timestep(model.num_time_steps - 1)
            curr_time = scp.LE('current_time_stamp', uncertain)
            (nc_data, mb) = NetCDFOutput.read_data(file_, curr_time)
            assert curr_time == nc_data['current_time_stamp'].item()
            assert scp.LE('mass_balance', uncertain) == mb

            # again, check that last time step
            (data_by_index, mb) = NetCDFOutput.read_data(file_, index=ix)
            assert curr_time == data_by_index['current_time_stamp'].item()
            assert scp.LE('mass_balance', uncertain) == mb

            with pytest.raises(IndexError):
                # check that no more data exists in NetCDF
                NetCDFOutput.read_data(file_, index=ix + 1)

        """ at least one matching time found """
        print(('All expected timestamps in {0} for output_ts_factor: {1}'
               .format(os.path.split(file_)[1], output_ts_factor)))

        # 2nd time around, look at uncertain filename so toggle uncertain flag
        uncertain = True

    # add this back in so cleanup script deletes the generated *.nc files
    model.outputters += o_put


def test_serialize_deserialize(output_filename):
    '''
    todo: this behaves in unexpected ways when using the 'model' testfixture.
    For now, define a model in here for the testing - not sure where the
    problem lies
    '''
    s_time = datetime(2014, 1, 1, 1, 1, 1)
    model = Model(start_time=s_time)
    model.spills += surface_point_line_spill(num_elements=5,
                                             start_position=(0, 0, 0),
                                             release_time=model.start_time)

    o_put = NetCDFOutput(output_filename)
    model.outputters += o_put
    model.movers += RandomMover(diffusion_coef=100000)

    # ==========================================================================
    # o_put = [model.outputters[outputter.id]
    #          for outputter in model.outputters
    #          if isinstance(outputter, NetCDFOutput)][0]
    # ==========================================================================

    model.rewind()
    print("step: {0}, _start_idx: {1}".format(-1, o_put._start_idx))
    for ix in range(2):
        model.step()
        print("step: {0}, _start_idx: {1}".format(ix, o_put._start_idx))

    o_put2 = NetCDFOutput.deserialize(o_put.serialize())
    assert o_put == o_put2
#     assert o_put._start_idx != o_put2._start_idx
#     assert o_put._middle_of_run != o_put2._middle_of_run
#     assert o_put != o_put2

    if os.path.exists(o_put.filename):
        print('\n{0} exists'.format(o_put.filename))


@pytest.mark.slow
def test_var_attr_spill_num(output_filename):
    '''
    call prepare_for_model_run and ensure the spill_num attributes are written
    correctly. Just a simple test that creates two models and adds a spill to
    each. It then runs both models and checks that the correct spill_name
    exists in 'spills_map' attribute for NetCDF output variable 'spill_num'
    '''
    def _make_run_model(spill, nc_name):
        'internal function'
        release_time = spill.release.release_time

        m = Model(start_time=release_time)
        m.outputters += NetCDFOutput(nc_name)
        m.spills += spill

        _run_model(m)
        return m

    def _del_nc_file(nc_name):
        try:
            os.remove(nc_name)
        except Exception:
            pass

    here = os.path.dirname(__file__)
    spills = []
    model = []
    nc_name = []

    for ix in (0, 1):
        spills.append(Spill(Release(datetime.now()),
                            name='m{0}_spill'.format(ix)))
        nc_name.append(os.path.join(here, 'temp_m{0}.nc'.format(ix)))

        _del_nc_file(nc_name[ix])
        _make_run_model(spills[ix], nc_name[ix])

    # open and check the correct spill_name exists in each netcdf file
    for ix, f_ in enumerate(nc_name):
        with nc.Dataset(f_) as data:
            assert (spills[ix].name in data.variables['spill_num'].spills_map)

            if ix == 0:
                assert (spills[1].name not in
                        data.variables['spill_num'].spills_map)

            if ix == 1:
                assert (spills[0].name not in
                        data.variables['spill_num'].spills_map)

            _del_nc_file(nc_name[ix])


def test_surface_concentration_output(model):
    """
    make sure the surface concentration is being computed and output

    Rewind model defined by model fixture.

    invoke model.step() till model runs all 5 steps

    For each step, make sure the surface_concentration data is there.
    """
    model.rewind()
    o_put = model.outputters[0]

    # FIXME:
    # o_put.surface_conc = "kde" # it's now default -- that should change!
    _run_model(model)

    file_ = o_put.filename
    with nc.Dataset(file_) as data:
        dv = data.variables
        for _step in range(model.num_time_steps):
            surface_conc = dv['surface_concentration']
            # FIXME -- maybe should test something more robust...
            assert not np.all(surface_conc[:] == 0.0)


def _run_model(model):
    'helper function'
    while True:
        try:
            model.step()
        except StopIteration:
            break
