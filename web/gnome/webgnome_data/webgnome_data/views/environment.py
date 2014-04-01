"""
Views for the Environment objects.
This currently includes Wind and Tide objects.
"""
import json

from pyramid.httpexceptions import (HTTPNotImplemented,
                                    HTTPUnsupportedMediaType)
from cornice import Service

from .helpers import (JSONImplementsOneOf,
                      ObjectImplementsOneOf,
                      UpdateObject, CreateObject)

env = Service(name='environment', path='/environment*obj_id',
              description="Environment API")

implemented_types = ('gnome.environment.Tide',
                     'gnome.environment.Wind',
                     )


@env.put()
def create_environment(request):
    '''Creates a new Environment object.'''
    json_request = json.loads(request.body)

    if not JSONImplementsOneOf(json_request, implemented_types):
        raise HTTPNotImplemented()

    # the pyramid URL parser returns a tuple of 0 or more
    # matching items, at least when using the * wild card
    obj_id = request.matchdict.get('obj_id')
    obj_id = obj_id[0] if obj_id else None
    print 'Our object ID:', obj_id

    obj = get_session_object(obj_id, request.session)
    if obj:
        if ObjectImplementsOneOf(obj, implemented_types):
            UpdateObject(obj, json_request)
        else:
            raise HTTPUnsupportedMediaType()
    else:
        obj = CreateObject(json_request)
        pass

    print 'our new timeseries: ', (obj.timeseries,)
    print 'our new ossm.timeseries: ', (obj.ossm.timeseries,)
    set_session_object(obj, request.session)
    return obj.serialize()


def get_session_object(obj_id, session):
    if 'objects' in session and obj_id in session['objects']:
        return session['objects'][obj_id]
    else:
        return None


def set_session_object(obj, session):
    if not 'objects' in session:
        session['objects'] = {}

    try:
        session['objects'][obj.id] = obj
    except AttributeError:
        session['objects'][id(obj)] = obj

    session.changed()


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
