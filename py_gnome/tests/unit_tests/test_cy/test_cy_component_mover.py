import os

import numpy as np

import gnome
from gnome import basic_types
from gnome.cy_gnome import cy_component_mover, cy_ossm_time
from gnome.environment import Wind
import cy_fixtures

from gnome.utilities.remote_data import get_datafile

datadir = os.path.join(os.path.dirname(__file__), r"sample_data")


class ComponentMove(cy_fixtures.CyTestMove):

    """
    Contains one test method to do one forecast move and one uncertainty move
    and verify that they are different

    Primarily just checks that CyComponentMover can be initialized correctly
    and all methods are invoked
    """

    def __init__(self):
        wind_file = get_datafile(os.path.join(datadir,
                             r"delaware_bay/ConstantWind.WND"))
        #self.ossm = cy_ossm_time.CyOSSMTime(wind_file,file_contains=basic_types.ts_format.magnitude_direction)
        wm = Wind(filename=wind_file)

        cats1_file = get_datafile(os.path.join(datadir,
                                r"delaware_bay/NW30ktwinds.cur"))
        cats2_file = get_datafile(os.path.join(datadir,
                                r"delaware_bay/SW30ktwinds.cur"))
        self.component = cy_component_mover.CyComponentMover()
        self.component.set_ossm(wm.ossm)
        #self.component.set_ossm(self.ossm)
        self.component.text_read(cats1_file,cats2_file)
        self.component.ref_point = (-75.262319, 39.142987)

        super(ComponentMove, self).__init__()
        self.ref[:] = (-75.262319, 39.142987, 0)
        self.component.prepare_for_model_run()
        self.component.prepare_for_model_step(self.model_time,
                self.time_step, 1, self.spill_size)

    def certain_move(self):
        """
        get_move for uncertain LEs
        """

        self.component.get_move(
            self.model_time,
            self.time_step,
            self.ref,
            self.delta,
            self.status,
            basic_types.spill_type.forecast,
            )
        print
        print self.delta

    def uncertain_move(self):
        """
        get_move for forecast LEs
        """

        self.component.get_move(
            self.model_time,
            self.time_step,
            self.ref,
            self.u_delta,
            self.status,
            basic_types.spill_type.uncertainty,
            )
        print
        print self.u_delta


def test_move():
    """
    test get_move for forecast and uncertain LEs
    """

    print
    print '--------------'
    print 'test certain_move and uncertain_move are different'
    tgt = ComponentMove()
    tgt.certain_move()
    tgt.uncertain_move()
    tgt.component.model_step_is_done()

    assert np.all(tgt.delta['lat'] != tgt.u_delta['lat'])
    assert np.all(tgt.delta['long'] != tgt.u_delta['long'])
    assert np.all(tgt.delta['z'] == tgt.u_delta['z'])


def test_certain_move():
    """
    test get_move for forecast LEs
    """

    print
    print '--------------'
    print 'test_certain_move'
    tgt = ComponentMove()
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

    print
    print '--------------'
    print 'test_uncertain_move'
    tgt = ComponentMove()
    tgt.uncertain_move()

    assert np.all(tgt.u_delta['lat'] != 0)
    assert np.all(tgt.u_delta['long'] != 0)
    assert np.all(tgt.u_delta['z'] == 0)


c_component = cy_component_mover.CyComponentMover()


def test_default_props():
    """
    test default properties
    """

    assert c_component.pat1_speed == 10
    assert c_component.pat1_angle == 0


def test_pat1_angle():
    """
    test setting / getting properties
    """

    c_component.pat1_angle = 10
    print c_component.pat1_angle
    assert c_component.pat1_angle == 10


def test_pat1_speed():
    """
    test setting / getting properties
    """

    c_component.pat1_speed = 5
    print c_component.pat1_speed
    assert c_component.pat1_speed == 5


def test_ref_point():
    """
    test setting / getting properties
    """

    tgt = (1, 2)
    c_component.ref_point = tgt  # can be a list or a tuple
    assert c_component.ref_point == tuple(tgt)
    c_component.ref_point = list(tgt)  # can be a list or a tuple
    assert c_component.ref_point == tuple(tgt)


if __name__ == '__main__':
    test_move()
