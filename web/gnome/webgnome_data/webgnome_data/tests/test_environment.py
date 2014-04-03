"""
Functional tests for the Gnome Environment object Web API
These include (Wind, Tide, etc.)
"""
from base import FunctionalTestBase


class WindTests(FunctionalTestBase):
    '''
        Tests out the Gnome Wind object API
    '''
    req_data = {'obj_type': 'gnome.environment.Wind',
                'description': u'Wind Object',
                'updated_at': '2014-03-26T14:52:45.385126',
                'source_type': u'undefined',
                'source_id': u'undefined',
                'timeseries': [('2012-11-06T20:10:30', 1.0, 0.0),
                               ('2012-11-06T20:11:30', 1.0, 45.0),
                               ('2012-11-06T20:12:30', 1.0, 90.0),
                               ('2012-11-06T20:13:30', 1.0, 120.0),
                               ('2012-11-06T20:14:30', 1.0, 180.0),
                               ('2012-11-06T20:15:30', 1.0, 270.0)],
                'units': u'meter per second'
                }

    def test_get_no_id(self):
        self.testapp.get('/environment', status=404)

    def test_get_invalid_id(self):
        obj_id = 0xdeadbeef
        self.testapp.get('/environment/{0}'.format(obj_id), status=404)

    def test_get_valid_id(self):
        # TODO: the strategy is to do a put with no id, which will create
        #       the object on the server and return the id.  We will then
        #       use the valid id to get the object.
        print 'Not implemented'

    def test_put_no_id(self):
        #print '\n\nEnvironment Put Request payload: {0}'.format(self.req_data)
        resp = self.testapp.put_json('/environment', params=self.req_data)

        # TODO: This should be working, but we need to put some asserts
        #       in here to validate what we are getting
        #print ('\n\nEnvironment Put Response payload: '
        #       '{0}'.format(resp.json_body))
        assert 'id' in resp.json_body
        assert 'obj_type' in resp.json_body
        assert 'timeseries' in resp.json_body

    def test_put_invalid_id(self):
        obj_id = 0xdeadbeef

        #print '\n\nEnvironment Put Request payload: {0}'.format(self.req_data)
        resp = self.testapp.put_json('/environment/{0}'.format(obj_id),
                                     params=self.req_data)

        # TODO: This should be working, but we need to put some asserts
        #       in here to validate what we are getting
        assert 'id' in resp.json_body
        assert 'obj_type' in resp.json_body
        assert 'timeseries' in resp.json_body

    def test_put_valid_id(self):
        print 'Not implemented'

        # TODO: the strategy is to do a put with no id, which will create
        #       the object on the server and return the id.  We will then
        #       use the valid id to update something on the object.


class TideTests(WindTests):
    '''
        Tests out the Gnome Tide object API
    '''
    req_data = {'obj_type': 'gnome.environment.Tide',
                'timeseries': [('2012-11-06T20:10:30', 1.0, 0.0),
                               ('2012-11-06T20:11:30', 1.0, 45.0),
                               ('2012-11-06T20:12:30', 1.0, 90.0),
                               ('2012-11-06T20:13:30', 1.0, 120.0),
                               ('2012-11-06T20:14:30', 1.0, 180.0),
                               ('2012-11-06T20:15:30', 1.0, 270.0)],
                }
