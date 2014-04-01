"""
Functional tests for the Mover Web API
"""
from base import FunctionalTestBase


class SimpleMoverTests(FunctionalTestBase):
    '''
        Tests out the Gnome Simple Mover API
    '''
    req_data = {'obj_type': 'gnome.movers.simple_mover.SimpleMover',
                'active_start': '-inf',
                'active_stop': 'inf',
                'on': True,
                'uncertainty_scale': 0.5,
                'velocity': (1.0, 1.0, 1.0)
                }

    def test_get_no_id(self):
        self.testapp.get('/mover', status=404)

    def test_get_invalid_id(self):
        obj_id = 0xdeadbeef
        self.testapp.get('/mover/{0}'.format(obj_id), status=404)

    def test_get_valid_id(self):
        # TODO: the strategy is to do a put with no id, which will create
        #       the object on the server and return the id.  We will then
        #       use the valid id to get the object.
        print 'Not implemented'

    def test_put_no_id(self):
        #print '\n\nMover Put Request payload: {0}'.format(self.req_data)
        resp = self.testapp.put_json('/mover', params=self.req_data)

        # TODO: This should be working, but we need to put some asserts
        #       in here to validate what we are getting

    def test_put_invalid_id(self):
        obj_id = 0xdeadbeef

        #print '\n\nMover Put Request payload: {0}'.format(self.req_data)
        resp = self.testapp.put_json('/mover/{0}'.format(obj_id),
                                     params=self.req_data)

        # TODO: This should be working, but we need to put some asserts
        #       in here to validate what we are getting

    def test_put_valid_id(self):
        print 'Not implemented'

        # TODO: the strategy is to do a put with no id, which will create
        #       the object on the server and return the id.  We will then
        #       use the valid id to update something on the object.


class WindMoverTests(SimpleMoverTests):
    '''
        Tests out the Gnome Wind Mover API
    '''
    req_data = {'obj_type': 'gnome.movers.wind_movers.WindMover',
                'active_start': '-inf',
                'active_stop': 'inf',
                'on': True,
                'uncertain_angle_scale': 0.4,
                'uncertain_angle_units': u'rad',
                'uncertain_duration': 3.0,
                'uncertain_speed_scale': 2.0,
                'uncertain_time_delay': 0.0,
                'wind': {'description': u'Wind Object',
                         'source_id': u'undefined',
                         'source_type': u'undefined',
                         'timeseries': [('2014-03-31T16:01:13', 5.0, 45.0)],
                         'units': u'm/s',
                         'updated_at': '2014-03-31T16:02:35.530359'}
                }
