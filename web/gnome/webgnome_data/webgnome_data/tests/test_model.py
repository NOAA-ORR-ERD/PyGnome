"""
Functional tests for the Model Web API
"""
from datetime import datetime

from base import FunctionalTestBase


class ModelTests(FunctionalTestBase):
    req_data = {'obj_type': u'gnome.model.Model',
                'cache_enabled': False,
                'duration': 86400.0,
                'start_time': '2014-04-09T15:00:00',
                'time_step': 900.0,
                'uncertain': False,
                'weathering_substeps': 1,
                'environment': [],
                'movers': [],
                'outputters': [],
                'spills': [],
                'weatherers': [],
                }

    def test_get_model_no_id(self):
        self.testapp.get('/model', status=404)

    def test_get_model_invalid_id(self):
        obj_id = 0xdeadbeef
        self.testapp.get('/model/{0}'.format(obj_id), status=404)

    def test_get_model_no_id_active(self):
        '''
            Here we test the get with no ID, but where an active model
            is attached to the session.
        '''
        print 'Not Implemented'

    def test_get_model_invalid_id_active(self):
        '''
            Here we test the get with an invalid ID, but where an active model
            is attached to the session.
        '''
        print 'Not Implemented'

    def test_get_model_valid_id(self):
        print 'Not Implemented'

    def test_put_model_no_id(self):
        print '\n\nModel Put Request payload: {0}'.format(self.req_data)
        resp = self.testapp.put_json('/model', params=self.req_data)
        print '\nModel Put Response payload: {0}'.format(resp.json_body)

        # TODO: This should be working, but we need to put some asserts
        #       in here to validate what we are getting
