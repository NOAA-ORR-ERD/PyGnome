'''
Tests CyMover base class can be instantiated
and it does not crash the python interpretor.

It is more of an abstract base class that should not be instantiated;
however, since that is not possible in cython, this is just to ensure
it works.

The functions being tested do not produce any results.
'''




from pytest import raises

from gnome.cy_gnome import cy_mover

cm = cy_mover.CyMover()


def test_repr():
    repr(cm)
    assert True


def test_str():
    str(cm)
    assert True


def test_prepare_for_model_run():
    '''
        Since our CyMover has no encapsulated C++ mover object, we should not
        be able to run any mover functions
    '''
    with raises(OSError):
        cm.prepare_for_model_run()


def test_prepare_for_model_step():
    '''
        Since our CyMover has no encapsulated C++ mover object, we should not
        be able to run any mover functions
    '''
    with raises(OSError):
        cm.prepare_for_model_step(0, 0, 0)


def test_model_step_is_done():
    '''
        Since our CyMover has no encapsulated C++ mover object, we should not
        be able to run any mover functions
    '''
    with raises(OSError):
        cm.model_step_is_done()
