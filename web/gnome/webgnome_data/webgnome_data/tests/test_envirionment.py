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
                }

        print '\n\nWind Put Request payload: {0}'.format(data)
        resp = self.testapp.put_json('/environment/{0}'.format(obj_id),
                                     params=data)
        print '\nWind Put Response payload: {0}'.format(resp.json_body)
