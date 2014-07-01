#!/usr/bin/env python
'''
Tests for netcdf_outputter
'''

import os
from datetime import datetime, timedelta

import pytest
from pytest import raises

import numpy
np = numpy

import netCDF4 as nc

from gnome.spill.elements import floating
from gnome.spill import point_line_release_spill, Spill, Release

from gnome.movers import RandomMover
from gnome.outputters import NetCDFOutput
from gnome.model import Model
from gnome.persist import load

base_dir = os.path.dirname(__file__)


@pytest.fixture(scope='module')
def model(sample_model, request):
    """
    Use fixture model_surface_release_spill and add a few things to it for the
    test
    """
    model = sample_model['model']

    model.cache_enabled = True

    model.spills += point_line_release_spill(num_elements=5,
                        start_position=sample_model['release_start_pos'],
                        release_time=model.start_time,
                        end_release_time=model.start_time + model.duration,
                        element_type=floating(windage_persist=-1))

    model.movers += RandomMover(diffusion_coef=100000)

    model.outputters += NetCDFOutput(os.path.join(base_dir,
                                                  u'sample_model.nc'))

    model.rewind()

    def cleanup():
        'cleanup outputters was added to sample_model and delete files'

        print '\nCleaning up %s' % model
        o_put = None

        for outputter in model.outputters:
            # there should only be 1!
            #if isinstance(outputter, NetCDFOutput):
            o_put = model.outputters[outputter.id]

            if hasattr(o_put, 'netcdf_filename'):
                if os.path.exists(o_put.netcdf_filename):
                    os.remove(o_put.netcdf_filename)
                    print "deleted: {0}".format(o_put.netcdf_filename)

                if (o_put._u_netcdf_filename is not None
                    and os.path.exists(o_put._u_netcdf_filename)):
                    os.remove(o_put._u_netcdf_filename)
                    print "deleted: {0}".format(o_put._u_netcdf_filename)

    request.addfinalizer(cleanup)
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


def test_exceptions(model):
    t_file = os.path.join(base_dir, 'temp.nc')
    spill_pair = model.spills

    def _del_temp_file():
        # clean up temporary file from previous run
        try:
            print 'remove temporary file {0}'.format(t_file)
            os.remove(t_file)
        except:
            pass

    # begin tests
    _del_temp_file()
    netcdf = NetCDFOutput(t_file, which_data='all')

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

    with raises(ValueError):
        # raise error because file 'temp.nc' should exist after 1st call
        netcdf.prepare_for_model_run(model_start_time=datetime.now(),
                                     spills=spill_pair,
                                     num_time_steps=4)
        netcdf.prepare_for_model_run(model_start_time=datetime.now(),
                                     spills=spill_pair,
                                     num_time_steps=4)

    with raises(AttributeError):
        'cannot change after prepare_for_model_run has been called'
        netcdf.which_data = 'all'

    _del_temp_file()


def test_exceptions_middle_of_run(model):
    """
    Test attribute exceptions are called when changing parameters in middle of
    run for 'which_data' and 'netcdf_filename'
    """
    model.rewind()
    model.step()
    o_put = [model.outputters[outputter.id]
             for outputter in model.outputters
             if isinstance(outputter, NetCDFOutput)][0]

    assert o_put.middle_of_run

    with raises(AttributeError):
        o_put.netcdf_filename = 'test.nc'

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

    assert os.path.exists(o_put.netcdf_filename)

    if model.uncertain:
        assert os.path.exists(o_put._u_netcdf_filename)
    else:
        assert not os.path.exists(o_put._u_netcdf_filename)


def test_write_output_standard(model):
    """
    rewind model defined by model fixture.
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
    for file_ in (o_put.netcdf_filename, o_put._u_netcdf_filename):
        with nc.Dataset(file_) as data:
            dv = data.variables
            time_ = nc.num2date(dv['time'][:], dv['time'].units,
                                calendar=dv['time'].calendar)

            idx = np.cumsum((dv['particle_count'])[:])
            idx = np.insert(idx, 0, 0)  # add starting index of 0

            for step in range(model.num_time_steps):
                scp = model._cache.load_timestep(step)

                # check time

                assert scp.LE('current_time_stamp', uncertain) == time_[step]

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

            print 'data in model matches output in {0}'.format(file_)

        # 2nd time around, we are looking at uncertain filename so toggle
        # uncertain flag

        uncertain = True


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
    for file_ in (o_put.netcdf_filename, o_put._u_netcdf_filename):
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

                    if len(sc_arr.shape) == 1:
                        assert np.all(nc_var[idx[step]:idx[step + 1]]
                                      == sc_arr)
                    elif len(sc_arr.shape) == 2:
                        assert np.all(nc_var[idx[step]:idx[step + 1], :]
                                      == sc_arr)
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
        NetCDFOutput.read_data(o_put.netcdf_filename)


@pytest.mark.parametrize(("output_ts_factor", "use_time"),
                         [(1, True), (1, False),
                          (2.4, True), (2.4, False)])
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
    for file_ in (o_put.netcdf_filename, o_put._u_netcdf_filename):
        _found_a_matching_time = False

        for idx, step in enumerate(range(0, model.num_time_steps,
                                                    int(output_ts_factor))):
            scp = model._cache.load_timestep(step)
            curr_time = scp.LE('current_time_stamp', uncertain)
            if use_time:
                nc_data = NetCDFOutput.read_data(file_, curr_time)
            else:
                nc_data = NetCDFOutput.read_data(file_, index=idx)

            # check time
            if curr_time == nc_data['current_time_stamp'].item():
                _found_a_matching_time = True

                # check standard variables
                assert np.allclose(scp.LE('positions', uncertain),
                                   nc_data['positions'], rtol, atol)
                assert np.all(scp.LE('spill_num', uncertain)[:]
                              == nc_data['spill_num'])
                assert np.all(scp.LE('status_codes', uncertain)[:]
                              == nc_data['status_codes'])

                # flag variable is not currently set or checked

                if 'mass' in scp.LE_data:
                    assert np.all(scp.LE('mass', uncertain)[:]
                                  == nc_data['mass'])

                if 'age' in scp.LE_data:
                    assert np.all(scp.LE('age', uncertain)[:]
                                  == nc_data['age'])

        if _found_a_matching_time:
            """ at least one matching time found """
            print ('\ndata in model matches for output in \n{0} \nand'
                   ' output_ts_factor: {1}'.format(file_, output_ts_factor))

        # 2nd time around, look at uncertain filename so toggle uncertain flag

        uncertain = True


def test_read_all_arrays(model):
    """
    tests the data returned by read_data is correct when `which_data` flag is
    'all'.
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
    for file_ in (o_put.netcdf_filename, o_put._u_netcdf_filename):
        _found_a_matching_time = False
        for step in range(model.num_time_steps):
            scp = model._cache.load_timestep(step)
            curr_time = scp.LE('current_time_stamp', uncertain)

            nc_data = NetCDFOutput.read_data(file_, curr_time,
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
                    else:
                        if key not in ['last_water_positions',
                                       'next_positions']:
                            assert np.all(scp.LE(key, uncertain)[:]
                                          == nc_data[key])

        if _found_a_matching_time:
            print ('\ndata in model matches for output in \n{0}'.format(file_))

        # 2nd time around, look at uncertain filename so toggle uncertain flag
        uncertain = True


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
    o_put.output_timestep = timedelta(seconds=model.time_step *
                                      output_ts_factor)

    del model.outputters[o_put.id]  # remove from list of outputters

    _run_model(model)

    assert (not os.path.exists(o_put.netcdf_filename))
    if o_put._u_netcdf_filename:
        assert (not os.path.exists(o_put._u_netcdf_filename))

    # now write netcdf output
    o_put.write_output_post_run(model.start_time,
                                model.num_time_steps,
                                spills=model.spills,
                                cache=model._cache,
                                uncertain=model.uncertain)

    assert os.path.exists(o_put.netcdf_filename)
    if model.uncertain:
        assert os.path.exists(o_put._u_netcdf_filename)

    uncertain = False
    for file_ in (o_put.netcdf_filename, o_put._u_netcdf_filename):
        for step in range(model.num_time_steps, int(output_ts_factor)):
            print "step: {0}".format(step)
            scp = model._cache.load_timestep(step)
            curr_time = scp.LE('current_time_stamp', uncertain)
            nc_data = NetCDFOutput.read_data(file_, curr_time)
            assert curr_time == nc_data['current_time_stamp'].item()

        if o_put.output_last_step:
            scp = model._cache.load_timestep(model.num_time_steps - 1)
            curr_time = scp.LE('current_time_stamp', uncertain)
            nc_data = NetCDFOutput.read_data(file_, curr_time)
            assert curr_time == nc_data['current_time_stamp'].item()

        """ at least one matching time found """
        print ('\nAll expected timestamps are written out for'
                ' output_ts_factor: {1}'.format(file_, output_ts_factor))

        # 2nd time around, look at uncertain filename so toggle uncertain flag
        uncertain = True

    # add this back in so cleanup script deletes the generated *.nc files
    model.outputters += o_put


@pytest.mark.parametrize(("json_"), ['save', 'webapi'])
def test_serialize_deserialize(json_):
    '''
    todo: this behaves in unexpected ways when using the 'model' testfixture.
    For now, define a model in here for the testing - not sure where the
    problem lies
    '''
    s_time = datetime(2014, 1, 1, 1, 1, 1)
    model = Model(start_time=s_time)
    model.spills += point_line_release_spill(num_elements=5,
        start_position=(0, 0, 0),
        release_time=model.start_time)

    o_put = NetCDFOutput(os.path.join(base_dir, u'xtemp.nc'))
    model.outputters += o_put
    model.movers += RandomMover(diffusion_coef=100000)

    #==========================================================================
    # o_put = [model.outputters[outputter.id]
    #          for outputter in model.outputters
    #          if isinstance(outputter, NetCDFOutput)][0]
    #==========================================================================

    model.rewind()
    print "step: {0}, _start_idx: {1}".format(-1, o_put._start_idx)
    for ix in range(2):
        model.step()
        print "step: {0}, _start_idx: {1}".format(ix, o_put._start_idx)

    #for json_ in ('save', 'webapi'):
    dict_ = o_put.deserialize(o_put.serialize(json_))
    o_put2 = NetCDFOutput.new_from_dict(dict_)
    if json_ == 'save':
        assert o_put == o_put2
    else:
        # _start_idx and _middle_of_run should not match
        assert o_put._start_idx != o_put2._start_idx
        assert o_put._middle_of_run != o_put2._middle_of_run
        assert o_put != o_put2

    if os.path.exists(o_put.netcdf_filename):
        print '\n{0} exists'.format(o_put.netcdf_filename)


def test_var_attr_spill_num():
    '''
    call prepare_for_model_run and ensure the spill_num attributes are written
    correctly. Just a simple test that creates two models and adds a spill to
    each. It then runs both models and checks that the correct spill_name
    exists in 'spills_map' attribute for NetCDF output variable 'spill_num'
    '''
    def _make_run_model(spill, nc_name):
        'internal funtion'
        m = Model()
        m.outputters += NetCDFOutput(nc_name)
        m.spills += spill
        _run_model(m)
        return m

    def _del_nc_file(nc_name):
        try:
            os.remove(nc_name)
        except:
            pass

    spills = []
    model = []
    nc_name = []
    for ix in (0, 1):
        spills.append(Spill(Release(datetime.now()),
                                    name='m{0}_spill'.format(ix)))
        nc_name.append(os.path.join(base_dir, 'temp_m{0}.nc'.format(ix)))
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


def _run_model(model):
    'helper function'
    while True:
        try:
            model.step()
        except StopIteration:
            break
