import os

import numpy as np

import gnome
from gnome import basic_types
from gnome.cy_gnome import cy_cats_mover, cy_ossm_time, cy_shio_time
from . import cy_fixtures

from ..conftest import testdata


class CatsMove(cy_fixtures.CyTestMove):

    """
    Contains one test method to do one forecast move and one uncertainty move
    and verify that they are different

    Primarily just checks that CyCatsMover can be initialized correctly
    and all methods are invoked
    """

    def __init__(self):
        file_ = testdata['CatsMover']['tide']
        self.shio = cy_shio_time.CyShioTime(file_)
        cur_file = testdata['CatsMover']['curr']
        yeardata_path = os.path.join(os.path.dirname(gnome.__file__),
                                     'data/yeardata/')
        self.cats = cy_cats_mover.CyCatsMover()
        self.cats.set_shio(self.shio)
        self.cats.text_read(cur_file)
        self.shio.set_shio_yeardata_path(yeardata_path)

        super(CatsMove, self).__init__()
        self.ref[:] = (-72.5, 41.17, 0)
        self.cats.prepare_for_model_run()
        self.cats.prepare_for_model_step(self.model_time,
                                         self.time_step, 1, self.spill_size)

    def certain_move(self):
        """
        get_move for uncertain LEs
        """

        self.cats.get_move(
            self.model_time,
            self.time_step,
            self.ref,
            self.delta,
            self.status,
            basic_types.spill_type.forecast,
            )
        print()
        print(self.delta)

    def uncertain_move(self):
        """
        get_move for forecast LEs
        """

        self.cats.get_move(
            self.model_time,
            self.time_step,
            self.ref,
            self.u_delta,
            self.status,
            basic_types.spill_type.uncertainty,
            )
        print()
        print(self.u_delta)


def test_move():
    """
    test get_move for forecast and uncertain LEs
    """

    print()
    print('--------------')
    print('test certain_move and uncertain_move are different')
    tgt = CatsMove()
    tgt.certain_move()
    tgt.uncertain_move()
    tgt.cats.model_step_is_done()

    assert np.all(tgt.delta['lat'] != tgt.u_delta['lat'])
    assert np.all(tgt.delta['long'] != tgt.u_delta['long'])
    assert np.all(tgt.delta['z'] == tgt.u_delta['z'])


def test_certain_move():
    """
    test get_move for forecase LEs
    """

    print()
    print('--------------')
    print('test_certain_move')
    tgt = CatsMove()
    tgt.certain_move()

    assert np.all(tgt.delta['lat'] != 0)
    assert np.all(tgt.delta['long'] != 0)
    assert np.all(tgt.delta['z'] == 0)

    assert np.all((tgt.delta['lat'])[:] == tgt.delta['lat'][0])
    assert np.all((tgt.delta['long'])[:] == tgt.delta['long'][0])


def test_uncertain_move():
    """
    test get_move for uncertainty LEs
    """

    print()
    print('--------------')
    print('test_uncertain_move')
    tgt = CatsMove()
    tgt.uncertain_move()

    assert np.all(tgt.u_delta['lat'] != 0)
    assert np.all(tgt.u_delta['long'] != 0)
    assert np.all(tgt.u_delta['z'] == 0)


c_cats = cy_cats_mover.CyCatsMover()


def test_default_props():
    """
    test default properties
    """

    assert c_cats.scale_value == 1
    assert c_cats.uncertain_eddy_diffusion == 0
    assert c_cats.ref_point is None


def test_scale_value():
    """
    test setting / getting properties
    """

    c_cats.scale_value = 0
    print(c_cats.scale_value)
    assert c_cats.scale_value == 0


def test_ref_point():
    """
    test setting / getting properties
    """

    tgt = (1, 2, 3)
    c_cats.ref_point = tgt  # can be a list or a tuple
    assert c_cats.ref_point == tuple(tgt)
    c_cats.ref_point = list(tgt)  # can be a list or a tuple
    assert c_cats.ref_point == tuple(tgt)

# @pytest.mark.xfail(reason="component mover can't take negative integer")
def test_run_backwards():
    """
    test that a component mover can work running backwards.
    """
    tgt = CatsMove()

    # run forward first:
    tgt.cats.get_move(
            tgt.model_time,
            tgt.time_step,
            tgt.ref,
            tgt.delta,
            tgt.status,
            basic_types.spill_type.forecast,
            )
    front_deltas = tgt.delta.copy()
    tgt.cats.model_step_is_done()

    # now backward:
    tgt.cats.get_move(
            tgt.model_time,
            - tgt.time_step,
            tgt.ref,
            tgt.delta,
            tgt.status,
            basic_types.spill_type.forecast,
            )
    deltas = tgt.delta
    tgt.cats.model_step_is_done()

    # deltas should be all the same:
    for d in deltas:
        assert d == deltas[0]

    print(front_deltas)
    print(deltas)
    # They should be the negative of the forward values
    assert deltas['lat'][0] == - front_deltas['lat'][0]
    assert deltas['long'][0] == - front_deltas['long'][0]
    assert deltas['z'][0] == - front_deltas['z'][0]


if __name__ == '__main__':
    test_move()
