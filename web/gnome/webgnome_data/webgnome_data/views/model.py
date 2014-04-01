"""
Views for the Model object.
"""
import json

from cornice import Service

model = Service(name='model', path='/model*obj_id', description="Model API")


@model.get()
def get_model(request):
    '''Returns Model object in JSON.'''

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

    return {'Model': 'Get() View',
            'Model ID': obj_id}


@model.post()
def create_model(request):
    '''Creates a new Model object.'''

    return {'Model': 'Post() View',
            'Model body': json.loads(request.body)}
