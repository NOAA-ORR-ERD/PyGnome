"""
tests for utilities.convert
"""
from __future__ import print_function
from __future__ import division
from __future__ import absolute_import
from __future__ import unicode_literals


import pytest
from gnome.basic_types import ts_format
from gnome.utilities import convert


def test_tsformat_uv():
    assert convert.tsformat("uv") == ts_format.uv


def test_tsformat_r_theta():
    assert convert.tsformat("r_theta") == ts_format.r_theta


def test_tsformat_rminustheta():
    assert convert.tsformat("r-theta") == ts_format.r_theta


def test_tsformat_mag_dir():
    assert convert.tsformat("magnitude_direction") == ts_format.r_theta


def test_tsformat_error():
    with pytest.raises(ValueError):
        convert.tsformat("random_string")


def test_tsformat_int():
    i = ts_format.uv.value
    print(i)

    assert convert.tsformat(i) == ts_format.uv


def test_tsformat_enum():
    i = ts_format.uv.value

    assert convert.tsformat(ts_format.uv.value) == ts_format.uv

