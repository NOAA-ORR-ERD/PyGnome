#!/usr/bin/env python

from gnome.utilities import transforms
import numpy as np
import random
import pytest


# use test fixtures invalid_rq, wind_circ and wind_rand
# Defined in conftest.py
def test_exceptions(invalid_rq):
    with pytest.raises(ValueError):
        transforms.r_theta_to_uv_wind(invalid_rq['rq'][0])
        transforms.r_theta_to_uv_wind(invalid_rq['rq'][1])
        transforms.r_theta_to_uv_wind(invalid_rq['rq'][2])
        transforms.r_theta_to_uv_wind(invalid_rq['rq'][3])
    
atol = 1e-14
rtol = 0

inv_atol = 1e-10    # randomly generate (r,theta), apply transform+inverse and check result is within 1e-10

def test_r_theta_to_uv_wind(wind_circ):
    uv_out = transforms.r_theta_to_uv_wind(wind_circ['rq'])
    print "actual (u,v): "
    print uv_out
    print "computed (u,v): "
    print wind_circ['uv']
    assert np.allclose( uv_out, wind_circ['uv'], atol, rtol)
    
    
def test_uv_to_r_theta_wind(wind_circ):
    rq_out = transforms.uv_to_r_theta_wind(wind_circ['uv'])
    print "actual (r,theta): "
    print rq_out
    print "computed (r,theta): "
    print wind_circ['rq']
    assert np.allclose( rq_out, wind_circ['rq'], atol, rtol)
    
def test_wind_inverse(rq_rand):
    """
    randomly generates an (r,theta) and applies the transform to convert to (u,v), then back to (r,theta).
    It checks the result is accurate to within 10-10 absolute tolerance
    """
    rq_out = transforms.uv_to_r_theta_wind( transforms.r_theta_to_uv_wind(rq_rand['rq']))
    print "actual (r,theta): "
    print rq_rand['rq']
    print "computed (r,theta): "
    print rq_out
    assert np.allclose( rq_out, rq_rand['rq'], inv_atol, rtol)
    
def test_r_theta_to_uv_current(curr_circ):
    uv_out = transforms.r_theta_to_uv_current( curr_circ['rq'])
    assert np.allclose( uv_out, curr_circ['uv'], atol, rtol)
    
def test_uv_to_r_theta_current(curr_circ):
    rq_out = transforms.uv_to_r_theta_current( curr_circ['uv'])
    assert np.allclose( rq_out, curr_circ['rq'], atol, rtol)
    
def test_current_inverse(rq_rand):
    """
    randomly generates an (r,theta) and applies the transform to convert to (u,v), then back to (r,theta).
    It checks the result is accurate to within 10-10 tolerance
    """
    rq_out = transforms.uv_to_r_theta_current( transforms.r_theta_to_uv_current(rq_rand['rq']))
    print "actual (r,theta): "
    print rq_rand['rq']
    print "computed (r,theta): "
    print rq_out
    assert np.allclose(rq_out, rq_rand['rq'], inv_atol, rtol)
