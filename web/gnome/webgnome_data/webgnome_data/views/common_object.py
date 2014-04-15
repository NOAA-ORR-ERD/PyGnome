"""
Common Gnome object request handlers.
"""
import json

from pyramid.httpexceptions import (HTTPNotFound,
                                    HTTPNotImplemented,
                                    HTTPUnsupportedMediaType)

from .helpers import (JSONImplementsOneOf,
                      ObjectImplementsOneOf,
                      UpdateObject, CreateObject,
                      FQNamesToList,
                      PyClassFromName)


def get_object(request, implemented_types):
    '''Returns a Gnome object in JSON.'''
    obj_id = obj_id_from_url(request)
    if not obj_id:
        return get_specifications(request, implemented_types)
    else:
        obj = get_session_object(obj_id, request.session)
        if obj:
            if ObjectImplementsOneOf(obj, implemented_types):
                return obj.serialize()
            else:
                raise HTTPUnsupportedMediaType()
        else:
            raise HTTPNotFound()


def get_specifications(request, implemented_types):
    specs = {}
    for t in implemented_types:
        try:
            name, scope = FQNamesToList((t,))[0]
            cls = PyClassFromName(name, scope)
            if cls:
                spec = dict([(n, None)
                             for n in cls._state.get_names(['read', 'update'])
                             ])
                spec['obj_type'] = t
                specs[name] = spec
        except ValueError as e:
            # - I think for right now, we will just continue on to the
            #   next type.
            # - There could be some exceptions raised that are not handled
            #   here.
            #print 'failed to get class for {0}'.format(t)
            #print 'error: {0}'.format(e)
            raise
    return specs


def create_or_update_object(request, implemented_types):
    '''Creates or Updates a Gnome object.'''
    json_request = json.loads(request.body)

    if not JSONImplementsOneOf(json_request, implemented_types):
        raise HTTPNotImplemented()

    obj = get_session_object(obj_id_from_url(request), request.session)
    if obj:
        try:
            UpdateObject(obj, json_request)
        except ValueError as e:
            # TODO: We might want to log this message somewhere, as the
            # response is a bit vague
            raise HTTPUnsupportedMediaType(e)
    else:
        obj = CreateObject(json_request, request.session['objects'])

    set_session_object(obj, request.session)
    return obj.serialize()


def obj_id_from_url(request):
    # the pyramid URL parser returns a tuple of 0 or more
    # matching items, at least when using the * wild card
    obj_id = request.matchdict.get('obj_id')
    return obj_id[0] if obj_id else None


def init_session_objects(session):
    if not 'objects' in session:
        session['objects'] = {}
        session.changed()


def get_session_object(obj_id, session):
    init_session_objects(session)

    if obj_id in session['objects']:
        return session['objects'][obj_id]
    else:
        return None


def set_session_object(obj, session):
    init_session_objects(session)

    try:
        session['objects'][obj.id] = obj
    except AttributeError:
        session['objects'][id(obj)] = obj

    session.changed()
