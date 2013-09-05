#!/usr/bin/env python

"""
tests for InfDateTime: special datetime object with -inf and inf times
"""

import pytest

from gnome.utilities.inf_datetime import InfTime, MinusInfTime, \
    InfDateTime


def test_init():
    dt = InfDateTime(2012, 10, 20)

    assert True


def test_init_inf():
    dt = InfDateTime('inf')

    assert isinstance(dt, InfTime)


def test_init_minusinf():
    dt = InfDateTime('-inf')

    assert isinstance(dt, MinusInfTime)


def test_max_range():
    with pytest.raises(ValueError):
        dt = InfDateTime(2039, 10, 20)


def test_min_range():
    with pytest.raises(ValueError):
        dt = InfDateTime(1969, 10, 20)


def test_init_fail1():
    with pytest.raises(ValueError):
        dt = InfDateTime('some_stuff')


def test_init_fail2():
    with pytest.raises(TypeError):
        dt = InfDateTime(2012)


def test_init_fail3():
    with pytest.raises(TypeError):
        dt = InfDateTime(2012, 10)


def test_iso_format():
    dt = InfDateTime(2012, 10, 20)

    assert dt.isoformat() == '2012-10-20T00:00:00'


# tests for the InfTime object

def test_Inf_init():
    dt = InfTime()

    assert True


def test_Inf_greater():
    dt = InfTime()
    dt2 = InfTime()

    assert dt > InfDateTime(2012, 10, 20)
    assert dt > InfDateTime.max
    assert dt > InfDateTime.min
    assert not dt > dt2
    assert not dt2 > dt


def test_Inf_less():
    dt = InfTime()
    dt2 = InfTime()

    assert not dt < InfDateTime(2012, 10, 20)
    assert not dt < InfDateTime.max
    assert not dt < InfDateTime.min
    assert not dt < dt2
    assert not dt2 < dt


def test_Inf_equal():
    dt = InfTime()
    dt2 = InfTime()

    assert not dt == InfDateTime(2012, 10, 20)
    assert not dt == InfDateTime.max
    assert not dt == InfDateTime.min
    assert dt == dt2
    assert dt2 == dt


def test_Inf_not_equal():
    dt = InfTime()
    dt2 = InfTime()

    assert dt != InfDateTime(2012, 10, 20)
    assert dt != InfDateTime.max
    assert dt != InfDateTime.min
    assert not dt != dt2
    assert not dt2 != dt


def test_Inf_great_equal():
    dt = InfTime()
    dt2 = InfTime()

    assert dt >= InfDateTime(2012, 10, 20)
    assert dt >= InfDateTime.max
    assert dt >= InfDateTime.min
    assert dt >= dt2
    assert dt2 >= dt


def test_Inf_less_equal():
    dt = InfTime()
    dt2 = InfTime()

    assert not dt <= InfDateTime(2012, 10, 20)
    assert not dt <= InfDateTime.max
    assert not dt <= InfDateTime.min
    assert dt <= dt2
    assert dt2 <= dt


# tests for the MinusInfTime object

def test_MinusInf_init():
    dt = MinusInfTime()

    assert True


def test_MinusInf_greater():
    dt = MinusInfTime()
    dt2 = MinusInfTime()

    assert not dt > InfDateTime(2012, 10, 20)
    assert not dt > InfDateTime.max
    assert not dt > InfDateTime.min
    assert not dt > dt2
    assert not dt2 > dt


def test_MinusInf_less():
    dt = MinusInfTime()
    dt2 = MinusInfTime()

    assert dt < InfDateTime(2012, 10, 20)
    assert dt < InfDateTime.max
    assert dt < InfDateTime.min
    assert not dt < dt2
    assert not dt2 < dt


def test_MinusInf_equal():
    dt = MinusInfTime()
    dt2 = MinusInfTime()

    assert not dt == InfDateTime(2012, 10, 20)
    assert not dt == InfDateTime.max
    assert not dt == InfDateTime.min
    assert dt == dt2
    assert dt2 == dt


def test_MinusInf_not_equal():
    dt = MinusInfTime()
    dt2 = MinusInfTime()

    assert dt != InfDateTime(2012, 10, 20)
    assert dt != InfDateTime.max
    assert dt != InfDateTime.min
    assert not dt != dt2
    assert not dt2 != dt


def test_MinusInf_great_equal():
    dt = MinusInfTime()
    dt2 = MinusInfTime()

    assert not dt >= InfDateTime(2012, 10, 20)
    assert not dt >= InfDateTime.max
    assert not dt >= InfDateTime.min
    assert dt >= dt2
    assert dt2 >= dt


def test_MinusInf_less_equal():
    dt = MinusInfTime()
    dt2 = MinusInfTime()

    assert dt <= InfDateTime(2012, 10, 20)
    assert dt <= InfDateTime.max
    assert dt <= InfDateTime.min
    assert dt <= dt2
    assert dt2 <= dt


def test_negate():
    assert isinstance(-InfTime(), MinusInfTime)
    assert isinstance(-MinusInfTime(), InfTime)


def test_compare():
    assert InfTime() > MinusInfTime()
    assert InfTime() >= MinusInfTime()
    assert not InfTime() < MinusInfTime()
    assert not InfTime() <= MinusInfTime()
    assert MinusInfTime() < InfTime()
    assert MinusInfTime() <= InfTime()
    assert not MinusInfTime() > InfTime()
    assert not MinusInfTime() >= InfTime()


def test_isoformat():
    assert InfDateTime('inf').isoformat() == 'inf'
    assert InfDateTime('-inf').isoformat() == '-inf'
    assert InfDateTime(2013, 4, 20, 5, 23).isoformat() \
        == '2013-04-20T05:23:00'


