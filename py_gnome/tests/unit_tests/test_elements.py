'''
Test various element types available for the Spills
Element Types are very simple classes. They simply define the initializers.
These are also tested in the test_spill_container module since it allows for
more comprehensive testing
'''
from datetime import datetime, timedelta
import os

import pytest
from pytest import raises

import numpy
np = numpy

from hazpy import unit_conversion as uc
import gnome
from gnome import array_types
from gnome.spill.elements import (InitWindages,
                            InitMassFromSpillAmount,
                            InitRiseVelFromDist,
                            InitRiseVelFromDropletSizeFromDist,
                            floating,
                            floating_weathering,
                            ElementType,
                            plume)

from gnome.utilities.distributions import (NormalDistribution,
                                           LogNormalDistribution,
                                           WeibullDistribution)

from gnome.spill import Spill, Release
from oil_library import get_oil_props
from gnome.persist import load

from conftest import mock_append_data_arrays


""" Helper functions """
# first key in windages array must be 'windages' because test function:
# test_element_type_serialize_deserialize assumes this is the case
windages = {'windages': array_types.windages,
            'windage_range': array_types.windage_range,
            'windage_persist': array_types.windage_persist}

mass_array = {'mass': array_types.mass}

rise_vel_array = {'rise_vel': array_types.rise_vel}

rise_vel_diameter_array = {'rise_vel': array_types.rise_vel,
                           'droplet_diameter': array_types.droplet_diameter}
oil = 'ALAMO'
num_elems = 10


def assert_dataarray_shape_size(arr_types, data_arrays, num_released):
    for key, val in arr_types.iteritems():
        assert data_arrays[key].dtype == val.dtype
        assert data_arrays[key].shape == (num_released,) + val.shape


""" Initializers - following are used for parameterizing tests """
fcn_list = (InitWindages(),
            InitMassFromSpillAmount(),
            InitMassFromSpillAmount(),
            InitRiseVelFromDist(),
            InitRiseVelFromDist(distribution=NormalDistribution(mean=0,
                                                                sigma=0.1)),
            InitRiseVelFromDist(distribution=LogNormalDistribution(mean=0,
                                                                   sigma=0.1)),
            InitRiseVelFromDist(distribution=WeibullDistribution(alpha=1.8,
                                                                 lambda_=(1 / (.693 ** (1 / 1.8)))
                                                                 )),
            InitRiseVelFromDropletSizeFromDist(NormalDistribution(mean=0,
                                                                  sigma=0.1))
            )

arrays_ = (windages,
           mass_array, mass_array,
           rise_vel_array, rise_vel_array, rise_vel_array, rise_vel_array,
           rise_vel_diameter_array)

spill_list = (None,
              Spill(Release(datetime.now()), amount=10, units='l'),
              Spill(Release(datetime.now()), amount=10, units='kg'),
              None, None, None, None,
              Spill(Release(datetime.now())))


@pytest.mark.parametrize(("fcn", "arr_types", "spill"),
                         zip(fcn_list, arrays_, spill_list))
def test_correct_particles_set_by_initializers(fcn, arr_types, spill):
    '''
    Tests that the correct elements (ones that
    were released last) are initialized
    '''
    # let's only set the values for the last 10 elements
    # this is not how it would be used, but this is just to make sure
    # the values for the correct elements are set
    data_arrays = mock_append_data_arrays(arr_types, num_elems)
    data_arrays = mock_append_data_arrays(arr_types, num_elems, data_arrays)
    substance = get_oil_props('oil_conservative')

    if spill is not None:
        spill.release.num_elements = 10

    fcn.initialize(num_elems, spill, data_arrays, substance)

    assert_dataarray_shape_size(arr_types, data_arrays, num_elems * 2)

    # contrived example since particles will be initialized for every timestep
    # when they are released. But just to make sure that only values for the
    # latest released elements are set
    for key in data_arrays:
        assert np.all(0 == data_arrays[key][:num_elems])

        # values for these particles should be initialized to non-zero
        assert np.any(0 != data_arrays[key][-num_elems:])


@pytest.mark.parametrize("fcn", fcn_list)
def test_element_type_serialize_deserialize(fcn):
    '''
    test serialization/deserialization of ElementType for various initiailzers
    '''
    element_type = ElementType(initializers=[fcn], substance=oil)

    json_ = element_type.serialize('save')
    dict_ = element_type.deserialize(json_)
    element_type2 = ElementType.new_from_dict(dict_)

    assert element_type == element_type2


class TestInitConstantWindageRange:
    @pytest.mark.parametrize(("fcn", "array"),
                             [(InitWindages(), windages),
                              (InitWindages([0.02, 0.03]), windages),
                              (InitWindages(), windages),
                              (InitWindages(windage_persist=-1), windages)])
    def test_initailize_InitConstantWindageRange(self, fcn, array):
        'tests initialize method'
        data_arrays = mock_append_data_arrays(array, num_elems)
        fcn.initialize(num_elems, None, data_arrays)
        assert_dataarray_shape_size(array, data_arrays, num_elems)

        assert np.all(data_arrays['windage_range'] == fcn.windage_range)
        assert np.all(data_arrays['windage_persist'] == fcn.windage_persist)

        np.all(data_arrays['windages'] != 0)
        np.all(data_arrays['windages'] >= data_arrays['windage_range'][:, 0])
        np.all(data_arrays['windages'] <= data_arrays['windage_range'][:, 1])

    def test_exceptions(self):
        bad_wr = [-1, 0]
        bad_wp = 0
        obj = InitWindages()
        with raises(ValueError):
            InitWindages(windage_range=bad_wr)

        with raises(ValueError):
            InitWindages(windage_persist=bad_wp)

        with raises(ValueError):
            obj.windage_range = bad_wr

        with raises(ValueError):
            obj.windage_persist = bad_wp


def test_initailize_InitMassFromSpillAmount():
    data_arrays = mock_append_data_arrays(mass_array, num_elems)
    substance = get_oil_props('oil_conservative')

    spill = Spill(Release(datetime.now()))
    spill.release.num_elements = 10
    spill.amount = num_elems
    spill.units = 'g'

    fcn = InitMassFromSpillAmount()
    fcn.initialize(num_elems, spill, data_arrays, substance)

    assert_dataarray_shape_size(mass_array, data_arrays, num_elems)
    mass_per_le = spill.get_mass('kg')/spill.release.num_elements
    assert np.all(mass_per_le == data_arrays['mass'])


def test_initialize_InitRiseVelFromDist_uniform():
    'Test initialize data_arrays with uniform dist'
    data_arrays = mock_append_data_arrays(rise_vel_array, num_elems)

    fcn = InitRiseVelFromDist()
    fcn.initialize(num_elems, None, data_arrays)

    assert_dataarray_shape_size(rise_vel_array, data_arrays, num_elems)

    assert np.all(0 != data_arrays['rise_vel'])
    assert np.all(data_arrays['rise_vel'] <= 1)
    assert np.all(data_arrays['rise_vel'] >= 0)


def test_initialize_InitRiseVelFromDropletDist_weibull():
    'Test initialize data_arrays with Weibull dist'
    num_elems = 10
    data_arrays = mock_append_data_arrays(rise_vel_diameter_array, num_elems)
    substance = get_oil_props('oil_conservative')
    spill = Spill(Release(datetime.now()))

    # (.001*.2) / (.693 ** (1 / 1.8)) - smaller droplet test case, in mm
    #                                   so multiply by .001
    dist = WeibullDistribution(alpha=1.8, lambda_=.000248)
    fcn = InitRiseVelFromDropletSizeFromDist(dist)
    fcn.initialize(num_elems, spill, data_arrays, substance)

    assert_dataarray_shape_size(rise_vel_array, data_arrays, num_elems)

    assert np.all(0 != data_arrays['rise_vel'])
    assert np.all(0 != data_arrays['droplet_diameter'])


def test_initialize_InitRiseVelFromDropletDist_weibull_with_min_max():
    'Test initialize data_arrays with Weibull dist'
    num_elems = 1000
    data_arrays = mock_append_data_arrays(rise_vel_diameter_array, num_elems)
    substance = get_oil_props('oil_conservative')
    spill = Spill(Release(datetime.now()))

    # (.001*3.8) / (.693 ** (1 / 1.8)) - larger droplet test case, in mm
    #                                    so multiply by .001
    dist = WeibullDistribution(min_=0.002, max_=0.004,
                               alpha=1.8, lambda_=.00456)
    fcn = InitRiseVelFromDropletSizeFromDist(dist)
    fcn.initialize(num_elems, spill, data_arrays, substance)

    # test for the larger droplet case above
    assert np.all(data_arrays['droplet_diameter'] >= .002)

    # test for the larger droplet case above
    assert np.all(data_arrays['droplet_diameter'] <= .004)


def test_initialize_InitRiseVelFromDist_normal():
    """
    test initialize data_arrays with normal dist
    assume normal distribution works fine - so statistics (mean, var) are not
    tested
    """
    num_elems = 1000
    data_arrays = mock_append_data_arrays(rise_vel_array, num_elems)

    dist = NormalDistribution(mean=0, sigma=0.1)
    fcn = InitRiseVelFromDist(distribution=dist)
    fcn.initialize(num_elems, None, data_arrays)

    assert_dataarray_shape_size(rise_vel_array, data_arrays, num_elems)

    assert np.all(0 != data_arrays['rise_vel'])


""" Element Types"""
# additional array_types corresponding with ElementTypes for following test
arr_types = {'windages': array_types.windages,
            'windage_range': array_types.windage_range,
            'windage_persist': array_types.windage_persist}

rise_vel = {'rise_vel': array_types.rise_vel}
rise_vel.update(arr_types)

mass_comp = {'mass_components': array_types.mass_components}
mass_comp.update(arr_types)
oil = 'ALAMO'

inp_params = [((floating(substance=oil),
                ElementType([InitWindages(), InitMassFromSpillAmount()],
                            substance=oil)), arr_types),
              ((floating(substance=oil),
                ElementType([InitWindages(), InitRiseVelFromDist()],
                            substance=oil)), rise_vel),
              ((floating(substance=oil),
                ElementType([InitMassFromSpillAmount(), InitRiseVelFromDist()],
                            substance=oil)), rise_vel),
              ((floating(substance=oil), floating_weathering(substance=oil)),
               mass_comp),
              ]


@pytest.mark.parametrize(("elem_type", "arr_types"), inp_params)
def test_element_types(elem_type, arr_types, sample_sc_no_uncertainty):
    """
    Tests data_arrays associated with the spill_container's
    initializers get initialized to non-zero values.
    Uses sample_sc_no_uncertainty fixture defined in conftest.py
    It initializes a SpillContainer object with two Spill objects. For first
    Spill object, set element_type=floating() and for the second Spill object,
    set element_type=elem_type[1] as defined in the tuple in inp_params
    """
    sc = sample_sc_no_uncertainty
    release_t = None

    for idx, spill in enumerate(sc.spills):
        spill.release.num_elements = 20
        spill.element_type = elem_type[idx]

        if release_t is None:
            release_t = spill.release.release_time

        # set release time based on earliest release spill
        if spill.release.release_time < release_t:
            release_t = spill.release.release_time

    time_step = 3600
    num_steps = 4
    sc.prepare_for_model_run(arr_types)

    for step in range(num_steps):
        current_time = release_t + timedelta(seconds=time_step * step)
        sc.release_elements(time_step, current_time)

        for spill in sc.spills:
            spill_mask = sc.get_spill_mask(spill)
            # todo: need better API for access
            s_arr_types = spill.get('array_types').keys()

            if 'mass_components' in s_arr_types:
                # floating_weathering() uses InitArraysFromOilProps() which
                # also sets the following arrays
                assert 'mass' in s_arr_types

            if np.any(spill_mask):
                for key in arr_types:
                    if key in s_arr_types:
                        assert np.all(sc[key][spill_mask] != 0)
                    else:
                        if sc.array_types[key].initial_value is not None:
                            assert np.all(sc[key][spill_mask] ==
                                          sc.array_types[key].initial_value)


@pytest.mark.parametrize(("fcn"), fcn_list)
def test_serialize_deserialize_initializers(fcn):
    for json_ in ('save', 'webapi'):
        cls = fcn.__class__
        dict_ = cls.deserialize(fcn.serialize(json_))

        if json_ == 'webapi':
            if 'distribution' in dict_:
                'webapi will replace dict with object so mock it here'
                dict_['distribution'] = eval(dict_['distribution']['obj_type']).new_from_dict(dict_['distribution'])

        n_obj = cls.new_from_dict(dict_)
        # following is requirement for 'save' files
        # if object has no read only parameters, then this is true for 'webapi'
        # as well
        assert n_obj == fcn


test_l = []
test_l.extend(fcn_list)
test_l.extend([ElementType(initializers=fcn) for fcn in fcn_list])
test_l.append(floating())


def test_serialize_deserialize():
    '''
    serialize/deserialize for 'save' option is tested in test_save_load
    This tests serialize/deserilize with 'webapi' option
    '''
    et = floating()
    et.initializers.append(InitMassFromSpillAmount())
    dict_ = ElementType.deserialize(et.serialize('webapi'))

    # for webapi, make new objects from nested objects before creating
    # new element_type
    dict_['initializers'] = et.initializers
    n_et = ElementType.new_from_dict(dict_)
    # following is not a requirement for webapi, but it is infact the case
    assert n_et == et


@pytest.mark.parametrize(("test_obj"), test_l)
def test_save_load(clean_temp, test_obj):
    '''
    test save/load for initializers and for ElementType objects containing
    each initializer. Tests serialize/deserialize as well.
    These are stored as nested objects in the Spill but this should also work
    so test it here
    '''
    refs = test_obj.save(clean_temp)
    test_obj2 = load(os.path.join(clean_temp, refs.reference(test_obj)))
    assert test_obj == test_obj2


@pytest.mark.parametrize(("substance_name", "density", "density_units"),
                         [("oil_conservative", None, 'kg/m^3'),
                          ("test_1", 1000.0, 'kg/m^3'),
                          ("test_2", 10.0, 'api')])
def test_plume_init(substance_name, density, density_units):
    'test the plume() helper creates oil_props object correctly'
    et = plume(substance_name=substance_name,
               density=density,
               density_units=density_units)
    assert substance_name == et.substance.name
    if density:
        assert np.isclose(et.substance.get_density(),
            uc.convert('density', density_units, 'kg/m^3', density))
        assert np.isclose(et.substance.api,
            uc.convert('density', density_units, 'api', density))


@pytest.mark.parametrize("substance", [u'ALAMO',
                                       51,  # oil record in DB
                                       get_oil_props(u'ALAMO')])
def test_element_type_init(substance):
    et = ElementType(substance=substance)
    if isinstance(substance, basestring):
        assert et.substance.get('name') == substance
    elif isinstance(substance, int):
        assert et.substance.get('id') == substance
    else:
        assert et.substance.get('name') == substance.get('name')
