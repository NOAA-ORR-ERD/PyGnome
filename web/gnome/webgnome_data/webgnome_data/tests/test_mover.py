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
        # 1. create the object by performing a put with no id
        # 2. get the valid id from the response
        # 3. perform an additional get of the object with a valid id
        # 4. check that our new JSON response matches the one from the create
        resp1 = self.testapp.put_json('/mover', params=self.req_data)

        obj_id = resp1.json_body['id']
        resp2 = self.testapp.get('/mover/{0}'.format(obj_id))

        assert resp2.json_body['id'] == obj_id
        assert resp2.json_body['obj_type'] == resp1.json_body['obj_type']
        assert resp2.json_body['active_start'] == resp1.json_body['active_start']
        assert resp2.json_body['active_stop'] == resp1.json_body['active_stop']
        assert resp2.json_body['velocity'] == resp1.json_body['velocity']

    def test_put_no_id(self):
        #print '\n\nMover Put Request payload: {0}'.format(self.req_data)
        resp = self.testapp.put_json('/mover', params=self.req_data)

        # Note: For this test, we just verify that an object with the right
        #       properties is returned.  We will validate the content in
        #       more elaborate tests.
        assert 'id' in resp.json_body
        assert 'obj_type' in resp.json_body
        assert 'active_start' in resp.json_body
        assert 'active_stop' in resp.json_body
        assert 'velocity' in resp.json_body

    def test_put_invalid_id(self):
        obj_id = 0xdeadbeef

        #print '\n\nMover Put Request payload: {0}'.format(self.req_data)
        resp = self.testapp.put_json('/mover/{0}'.format(obj_id),
                                     params=self.req_data)

        # Note: This test is very similar to a put with no ID, and has the same
        #       asserts.
        assert 'id' in resp.json_body
        assert 'obj_type' in resp.json_body
        assert 'active_start' in resp.json_body
        assert 'active_stop' in resp.json_body
        assert 'velocity' in resp.json_body

    def test_put_valid_id(self):
        # 1. create the object by performing a put with no id
        # 2. get the valid id from the response
        # 3. update the properties in the JSON response
        # 4. update the object by performing a put with a valid id
        # 5. check that our new properties are in the new JSON response
        resp = self.testapp.put_json('/mover', params=self.req_data)

        obj_id = resp.json_body['id']
        req_data = resp.json_body
        self.perform_updates(req_data)

        resp = self.testapp.put_json('/mover/{0}'.format(obj_id),
                                     params=req_data)
        self.check_updates(resp.json_body)

    def perform_updates(self, json_obj):
        '''
            We can overload this function when subclassing our tests
            for new object types.
        '''
        json_obj['velocity'] = [10.0, 10.0, 10.0]
        json_obj['on'] = False

    def check_updates(self, json_obj):
        '''
            We can overload this function when subclassing our tests
            for new object types.
        '''
        assert json_obj[u'velocity'] == [10.0, 10.0, 10.0]
        assert json_obj['on'] == False


class WindMoverTests(SimpleMoverTests):
    '''
        Tests out the Gnome Wind Mover API
    '''
    wind_req_data = {'obj_type': 'gnome.environment.Wind',
                     'description': u'Wind Object',
                     'updated_at': '2014-03-26T14:52:45.385126',
                     'source_type': u'undefined',
                     'source_id': u'undefined',
                     'timeseries': [('2012-11-06T20:10:30', (1.0, 0.0)),
                                    ('2012-11-06T20:11:30', (1.0, 45.0)),
                                    ('2012-11-06T20:12:30', (1.0, 90.0)),
                                    ('2012-11-06T20:13:30', (1.0, 120.0)),
                                    ('2012-11-06T20:14:30', (1.0, 180.0)),
                                    ('2012-11-06T20:15:30', (1.0, 270.0))],
                     'units': u'meter per second'
                     }

    req_data = {'obj_type': 'gnome.movers.wind_movers.WindMover',
                   'active_start': '-inf',
                   'active_stop': 'inf',
                   'on': True,
                   'uncertain_angle_scale': 0.4,
                   'uncertain_angle_units': u'rad',
                   'uncertain_duration': 3.0,
                   'uncertain_speed_scale': 2.0,
                   'uncertain_time_delay': 0.0,
                   'wind': None
                   }

    def get_wind_obj(self, req_data):
        resp = self.testapp.put_json('/environment', params=req_data)
        return resp.json_body

    def test_get_valid_id(self):
        print 'Not Implemented'

    def test_put_no_id(self):
        # WindMover reauires a valid Wind object for creation
        wind_obj = self.get_wind_obj(self.wind_req_data)

        self.req_data['wind'] = wind_obj
        resp = self.testapp.put_json('/mover', params=self.req_data)
        print 'resp.json_body', resp.json_body

        # Note: For this test, we just verify that an object with the right
        #       properties is returned.  We will validate the content in
        #       more elaborate tests.
        assert 'id' in resp.json_body
        assert 'obj_type' in resp.json_body
        assert 'active_start' in resp.json_body
        assert 'active_stop' in resp.json_body
        assert 'velocity' in resp.json_body

    def test_put_invalid_id(self):
        print 'Not Implemented'

    def test_put_valid_id(self):
        print 'Not Implemented'
