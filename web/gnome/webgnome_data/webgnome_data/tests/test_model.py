"""
Functional tests for the Model Web API
"""
from datetime import datetime

from base import FunctionalTestBase

mock_model = '''
{"obj_type": "gnome.model.Model",
 "map": "gnome.map.MapFromBNA",
 "uncertain": true,
 "weathering_substeps": 1,
 "duration": 172800.0,
 "time_step": 1800.0,
 "cache_enabled": false,
 "start_time": "2013-02-13T09:00:00",
 "outputters": {"dtype": "<class 'gnome.outputters.outputter.Outputter'>",
                "items": [["gnome.renderer.Renderer", "0"]]
                },
 "movers": {"dtype": "<class 'gnome.movers.movers.Mover'>",
            "items": [["gnome.movers.random_movers.RandomMover", "0"],
                      ["gnome.movers.wind_movers.WindMover", "1"],
                      ["gnome.movers.current_movers.CatsMover", "2"],
                      ["gnome.movers.current_movers.CatsMover", "3"],
                      ["gnome.movers.current_movers.CatsMover", "4"]]
            },
 "environment": {"dtype": "<class 'gnome.environment.Environment'>",
                 "items": [["gnome.environment.Wind", "0"],
                           ["gnome.environment.Tide", "1"],
                           ["gnome.environment.Tide", "2"]]
                 },
 "weatherers": {"dtype": "<class 'gnome.weatherers.core.Weatherer'>",
                "items": [["gnome.weatherers.core.Weatherer", "0"]]
                },
 "spills": {"certain_spills": {"dtype": "<class 'gnome.spill.Spill'>",
                               "items": [["gnome.spill.Spill", "0"]]
                               },
            "uncertain_spills": {"dtype": "<class 'gnome.spill.Spill'>",
                                 "items": [["gnome.spill.Spill", "0"]]
                                 }
            },
 }
'''


class ModelTests(FunctionalTestBase):
    req_data = {'obj_type': 'gnome.model.Model',
                'start_time': '2014-03-31T14:00:00',
                'duration': 86400.0,
                'time_step': 900.0,
                'weathering_substeps': 1,
                'cache_enabled': False,
                'uncertain': False,
                'map_id': None,
                'environment': [],
                'movers': [],
                'outputters': [],
                'certain_spills': [],
                'weatherers': [],
                }

    def test_get_model_no_id(self):
        self.testapp.get('/model', status=404)

    def test_get_model_invalid_id(self):
        obj_id = 0xdeadbeef
        self.testapp.get('/model/{0}'.format(obj_id), status=404)

    def test_get_model_valid_id(self):
        print '\n\nNot Implemented'

    def test_put_model_no_id(self):
        print '\n\nNot Implemented'

        # TODO: Model serialize/deserialize needs to be working for this
        #       Basically a proper format for the JSON request payload is
        #       not really finalized yet.
        #print '\n\nModel Put Request payload: {0}'.format(self.req_data)
        #resp = self.testapp.put_json('/model', params=self.req_data)
        #print '\nModel Put Response payload: {0}'.format(resp.json_body)

        # TODO: This should be working, but we need to put some asserts
        #       in here to validate what we are getting
