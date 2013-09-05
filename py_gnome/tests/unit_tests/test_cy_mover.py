'''
Tests CyMover base class can be instantiated
and it does not crash the python interpretor.

It is more of an abstract base class that should not be instantiated;
however, since that is not possible in cython, this is just to ensure
it works.

The functions being tested do not produce any results.
'''

from gnome.cy_gnome import cy_mover

cm = cy_mover.CyMover()


def test_repr():
    repr(cm)
    assert True


def test_str():
    str(cm)
    assert True


def test_prepare_for_model_run():
    cm.prepare_for_model_run()
    assert True


def test_prepare_for_model_step():
    """ give it dummy input - it should not do anything"""

    cm.prepare_for_model_step(0, 0, 0)
    assert True


def test_model_step_is_done():
    cm.model_step_is_done()
    assert True


