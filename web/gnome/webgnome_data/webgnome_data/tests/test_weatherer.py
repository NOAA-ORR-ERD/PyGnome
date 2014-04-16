"""
Functional tests for the Gnome Environment object Web API
These include (Wind, Tide, etc.)
"""
from datetime import datetime

from base import FunctionalTestBase


class WeathererTests(FunctionalTestBase):
    '''
        Tests out the Gnome Wind object API
    '''
    req_data = {'obj_type': u'gnome.weatherers.core.Weatherer',
                'json_': 'webapi',
                'id': u'b505b505-c0fe-11e3-b8f2-3c075404121a',
                'active_start': '-inf',
                'active_stop': 'inf',
                'on': True,
                }

    def test_get_no_id(self):
        resp = self.testapp.get('/weatherer')

        assert 'obj_type' in self.req_data
        obj_type = self.req_data['obj_type'].split('.')[-1]

        assert (obj_type, obj_type) in [(name, obj['obj_type'].split('.')[-1])
                            for name, obj in resp.json_body.iteritems()]

    def test_get_invalid_id(self):
        obj_id = 0xdeadbeef
        self.testapp.get('/weatherer/{0}'.format(obj_id), status=404)

    def test_get_valid_id(self):
        # 1. create the object by performing a put with no id
        # 2. get the valid id from the response
        # 3. perform an additional get of the object with a valid id
        # 4. check that our new JSON response matches the one from the create
        resp1 = self.testapp.put_json('/weatherer', params=self.req_data)

        obj_id = resp1.json_body['id']
        resp2 = self.testapp.get('/weatherer/{0}'.format(obj_id))

        assert resp2.json_body['id'] == obj_id
        assert resp2.json_body['obj_type'] == resp1.json_body['obj_type']
        assert resp2.json_body['active_start'] == resp1.json_body['active_start']
        assert resp2.json_body['on'] == resp1.json_body['on']

    def test_put_no_id(self):
        #print '\n\nEnvironment Put Request payload: {0}'.format(self.req_data)
        resp = self.testapp.put_json('/weatherer', params=self.req_data)

        # Note: For this test, we just verify that an object with the right
        #       properties is returned.  We will validate the content in
        #       more elaborate tests.
        assert 'id' in resp.json_body
        assert 'obj_type' in resp.json_body
        assert 'active_start' in resp.json_body
        assert 'on' in resp.json_body

    def test_put_invalid_id(self):
        obj_id = 0xdeadbeef

        #print '\n\nEnvironment Put Request payload: {0}'.format(self.req_data)
        resp = self.testapp.put_json('/weatherer/{0}'.format(obj_id),
                                     params=self.req_data)

        # Note: This test is very similar to a put with no ID, and has the same
        #       asserts.
        assert 'id' in resp.json_body
        assert 'obj_type' in resp.json_body
        assert 'active_start' in resp.json_body
        assert 'on' in resp.json_body

    def test_put_valid_id(self):
        # 1. create the object by performing a put with no id
        # 2. get the valid id from the response
        # 3. update the properties in the JSON response
        # 4. update the object by performing a put with a valid id
        # 5. check that our new properties are in the new JSON response
        resp = self.testapp.put_json('/weatherer', params=self.req_data)

        obj_id = resp.json_body['id']
        req_data = resp.json_body
        self.perform_updates(req_data)

        resp = self.testapp.put_json('/weatherer/{0}'.format(obj_id),
                                     params=req_data)
        self.check_updates(resp.json_body)

    def perform_updates(self, json_obj):
        '''
            We can overload this function when subclassing our tests
            for new object types.
        '''
        self.now = datetime.now().isoformat()
        json_obj['active_start'] = self.now
        json_obj['on'] = False

    def check_updates(self, json_obj):
        '''
            We can overload this function when subclassing our tests
            for new object types.
        '''
        assert json_obj['active_start'] == self.now
        assert json_obj['on'] == False
