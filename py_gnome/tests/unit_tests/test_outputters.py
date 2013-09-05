#!/usr/bin/env python

"""
test code for the Outputter classes


"""

import gnome


def test_base():
    """
    test the base class
    """

    outputter = gnome.outputter.Outputter()

    outputter.write_output(3)
    outputter.prepare_for_model_run()
    outputter.prepare_for_model_step()
    outputter.model_step_is_done()
    outputter.rewind()

    assert True


if __name__ == '__main__':
    pass
