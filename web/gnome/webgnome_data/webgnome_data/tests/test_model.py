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

mock_model = {"id": "10",
              "map": None,
              "uncertain": False,
              "time": {
                       "start": "{{ model start time }}",
                       "stop": "{{ model stop time }}",
                       "step": "{{ model step increment }}",
                       "hours": "{{ model duration hours }}",
                       "days": "{{ model duration days }}"
                       },
              "environments": [{"id": "{{ environment_id }}"},
                               {"id": "{{ environment_id }}"}],
              "movers": [{"id": "{{ mover_id }}"},
                         {"id": "{{ mover_id }}"}],
              "spills": [{"id": "{{ spill_id }}"},
                         {"id": "{{ spill_id }}"}]
              }


class ModelTests(FunctionalTestBase):
    def test_get_model(self):
        resp = self.testapp.get('/model')
        self.model_body = resp.json_body
        print resp
        pass

    def test_post_model(self):
        start = datetime(2012, 12, 1, 2, 30)
        data = {
            'start_time': start.isoformat(),
            'uncertain': True,
            'time_step': 200,
            'duration_days': 20,
            'duration_hours': 1
        }

        print '\n\nModel Post Request payload: {0}'.format(data)
        resp = self.testapp.post_json('/model', data)
        print '\nModel Post Response payload: {0}'.format(resp.json_body)
        pass
