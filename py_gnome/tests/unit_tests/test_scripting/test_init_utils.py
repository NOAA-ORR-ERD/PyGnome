"""
tests for init functions
"""


import gnome.scripting as gs


def test_constant_point_current_mover():
    """
    Make a pure east current at 1 m/s
    """
    cm = gs.constant_point_current_mover(1, 90)

    assert cm.current.u == 1.0
    assert cm.current.v == 0.0