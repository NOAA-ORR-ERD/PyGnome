"""
Views for the Environment objects.
This currently includes Wind and Tide objects.
"""
import json

from pyramid.httpexceptions import HTTPNotImplemented, HTTPConflict
from cornice import Service

from .helpers import JSONImplementsOneOf

env = Service(name='environment', path='/environment*obj_id',
              description="Environment API")

implemented_types = ('gnome.environment.Tide',
                     'gnome.environment.Wind',
                     )


@env.put()
def create_environment(request):
    '''Creates a new Model object.'''
    # if JSON payload does not implement one of (Wind, Tide):
    #     return 501 Not Implemented
    #
    # if the ID refers to an existing object (where do we look???):
    #     if existing object implements one of (Wind, Tide)
    #         update the object
    #     else:
    #         return 409 Conflict
    # else:
    #     create the object
    # return the object
    print 'request body:', json.loads(request.body)

    if not JSONImplementsOneOf(json.loads(request.body), implemented_types):
        raise HTTPNotImplemented()

    if request.matchdict['obj_id']:
        obj_id = request.matchdict['obj_id'][0]
    else:
        obj_id = None
    print 'Our object ID:', obj_id

    return {'Wind': 'Post() View',
            'Wind body': json.loads(request.body)}


@env.get()
def get_environment(request):
    '''Returns an Gnome Environment object in JSON.'''

    if request.matchdict['obj_id']:
        obj_id = request.matchdict['obj_id'][0]
    else:
        obj_id = None

    #print 'request.session:', request.session
    #if 'my_counter' in request.session:
    #    request.session['my_counter'] += 1
    #else:
    #    request.session['my_counter'] = 0
    #request.session.changed()

    return {'Wind': 'Get() View',
            'Wind ID': obj_id}


wind_create_request_payload = '''
{"obj_type": "gnome.environment.Wind",
 "id": "99991111"
 "name": "wind",
 "description": "Wind Object",
 "source_type": "undefined",
 "source_id": "undefined",
 "units": "m/s"
 "timeseries": [["2013-02-13T09:00:00", 5.0, 180.0],
                ["2013-02-14T03:00:00", 5.0, 180.0]],
 "updated_at": "2014-03-20T14:03:45.609367",
}
'''

wind_create_request_payload = '''
{
 "name": "Wind",
 "source_type": "undefined",
 "units": "knots",
 "timeseries": [["2014-03-20T14:46:48-07:00","34.76","337.22"]]
 "updated_at": null,
 }
'''

wind_create_response_payload = '''
{"obj_type": "webgnome.model_manager.WebWind",
 "name": "Wind Object",
 "description": "Wind Object",
 "source_type": "undefined",
 "source_id": "undefined",
 "units": "knots",
 "timeseries": [["2014-03-20T14:46:48", 34.76, 337.22]],
 "updated_at": "2014-03-20T14:46:48.586412",
 }
'''
