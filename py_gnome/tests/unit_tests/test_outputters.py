#!/usr/bin/env python

"""
test code for the Outputter classes


"""

from gnome.outputter import Outputter


def test_base():
    """
    test the partial functionality implemented in base class
    """

    outputter = Outputter()

    #==========================================================================
    # outputter.write_output(3)
    # outputter.prepare_for_model_run()
    # outputter.prepare_for_model_step()
    # outputter.model_step_is_done()
    # outputter.rewind()
    #==========================================================================

    assert True


if __name__ == '__main__':
    pass
