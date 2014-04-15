"""
Functional tests for the Gnome Release object Web API
"""
from base import FunctionalTestBase


class ReleaseTests(FunctionalTestBase):
    '''
        Tests out the Gnome Release object API
    '''
    req_data = {'obj_type': u'gnome.spill.release.Release',
                'json_': u'webapi',
                'num_elements': 0,
                'num_released': 0,
                'release_time': '2014-04-14T11:00:10.860521',
                'start_time_invalid': True,
                }

    def test_get_no_id(self):
        resp = self.testapp.get('/release')

        #print 'Our Response:', resp.json_body
        if hasattr(self, 'req_data'):
            obj_type = self.req_data['obj_type'].split('.')[-1]
            assert obj_type in [r['obj_type'].split('.')[-1]
                                for r in resp.json_body]

    def test_get_invalid_id(self):
        obj_id = 0xdeadbeef
        self.testapp.get('/release/{0}'.format(obj_id), status=404)

    def test_get_valid_id(self):
        # 1. create the object by performing a put with no id
        # 2. get the valid id from the response
        # 3. perform an additional get of the object with a valid id
        # 4. check that our new JSON response matches the one from the create
        resp1 = self.testapp.put_json('/release', params=self.req_data)

        obj_id = resp1.json_body['id']
        resp2 = self.testapp.get('/release/{0}'.format(obj_id))

        assert resp2.json_body['id'] == obj_id
        assert resp2.json_body['obj_type'] == resp1.json_body['obj_type']

    def test_put_no_id(self):
        #print '\n\nEnvironment Put Request payload: {0}'.format(self.req_data)
        resp = self.testapp.put_json('/release', params=self.req_data)

        # Note: For this test, we just verify that an object with the right
        #       properties is returned.  We will validate the content in
        #       more elaborate tests.
        assert 'id' in resp.json_body
        assert 'obj_type' in resp.json_body
        assert 'timeseries' in resp.json_body

    def test_put_invalid_id(self):
        obj_id = 0xdeadbeef

        #print '\n\nEnvironment Put Request payload: {0}'.format(self.req_data)
        resp = self.testapp.put_json('/release/{0}'.format(obj_id),
                                     params=self.req_data)

        # Note: This test is very similar to a put with no ID, and has the same
        #       asserts.
        assert 'id' in resp.json_body
        assert 'obj_type' in resp.json_body
        assert 'timeseries' in resp.json_body

    def test_put_valid_id(self):
        # 1. create the object by performing a put with no id
        # 2. get the valid id from the response
        # 3. update the properties in the JSON response
        # 4. update the object by performing a put with a valid id
        # 5. check that our new properties are in the new JSON response
        resp = self.testapp.put_json('/release', params=self.req_data)

        obj_id = resp.json_body['id']
        req_data = resp.json_body
        self.perform_updates(req_data)

        resp = self.testapp.put_json('/release/{0}'.format(obj_id),
                                     params=req_data)
        self.check_updates(resp.json_body)

    def perform_updates(self, json_obj):
        '''
            We can overload this function when subclassing our tests
            for new object types.
        '''
        json_obj['description'] = u'Wind Object (updated)'

    def check_updates(self, json_obj):
        '''
            We can overload this function when subclassing our tests
            for new object types.
        '''
        assert 'id' in json_obj
        assert 'obj_type' in json_obj
        assert 'description' in json_obj
        assert json_obj[u'description'] == u'Wind Object (updated)'
