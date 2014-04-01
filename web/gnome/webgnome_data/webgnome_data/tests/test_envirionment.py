"""
Functional tests for the Model Web API
"""
from base import FunctionalTestBase


class EnvironmentTests(FunctionalTestBase):
    '''
        Tests out the Gnome Environment objects (Wind, Tide, etc.)
    '''
    def test_get_wind_no_id(self):
        resp = self.testapp.get('/environment')

        self.model_body = resp.json_body
        print resp

    def test_get_wind_invalid_id(self):
        obj_id = 0xdeadbeef
        resp = self.testapp.get('/environment/{0}'.format(obj_id))

        self.model_body = resp.json_body
        print resp

    def test_get_wind(self):
        # we need to get the id of a valid object somehow.
        obj_id = 0xdeadbeef
        resp = self.testapp.get('/environment/{0}'.format(obj_id))

        self.model_body = resp.json_body
        print resp

    def test_put_wind_no_id(self):
        data = {'obj_type': 'gnome.environment.Wind',
                }

        print '\n\nWind Put Request payload: {0}'.format(data)
        resp = self.testapp.put_json('/environment', params=data)
        print '\nWind Put Response payload: {0}'.format(resp.json_body)

    def test_put_wind_invalid_id(self):
        obj_id = 0xdeadbeef
        data = {'obj_type': 'gnome.environment.Wind',
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

        print '\n\nWind Put Request payload: {0}'.format(data)
        resp = self.testapp.put_json('/environment/{0}'.format(obj_id),
                                     params=data)
        print '\nWind Put Response payload: {0}'.format(resp.json_body)
