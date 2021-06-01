"""
tests for time utilities
"""





from gnome.scripting import *


def test_seconds():
    five = seconds() * 900
    assert five.total_seconds() == 900


def test_seconds2():
    five = seconds(900)
    assert five.total_seconds() == 900


def test_minutes():
    five = minutes() * 5
    assert five.total_seconds() == 60 * 5


def test_minutes2():
    five = minutes(5)
    assert five.total_seconds() == 60 * 5


def test_hours():
    five = hours() * 5
    assert five.total_seconds() == 3600 * 5


def test_hours2():
    five = hours(5)
    assert five.total_seconds() == 3600 * 5


def test_days():
    five = days() * 3
    assert five.total_seconds() == 3600 * 24 * 3


def test_days2():
    five = days(3)
    assert five.total_seconds() == 3600 * 24 * 3


def test_days3():
    five = days(1.5)
    assert five.total_seconds() == 3600 * 24 * 1.5


def test_week():
    one_week = weeks()
    assert one_week == days(7)


def test_week2():
    two_weeks = weeks() * 2
    assert two_weeks == days() * 14


def test_addition():
    """
    This is really just testing timedelta, but jsut in case
    """
    span = days() + 12 * hours()
    assert span == hours(36)

