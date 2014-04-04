"""
Common Gnome object request handlers.
"""
import json

from pyramid.httpexceptions import (HTTPNotFound,
                                    HTTPNotImplemented,
                                    HTTPUnsupportedMediaType)

from .helpers import (JSONImplementsOneOf,
                      ObjectImplementsOneOf,
                      UpdateObject, CreateObject)


def get_object(request, implemented_types):
    '''Returns a Gnome object in JSON.'''
    obj = get_session_object(obj_id_from_url(request), request.session)
    if obj:
        if ObjectImplementsOneOf(obj, implemented_types):
            return obj.serialize()
        else:
            raise HTTPUnsupportedMediaType()
    else:
        raise HTTPNotFound()


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
        obj = CreateObject(json_request)

    set_session_object(obj, request.session)
    return obj.serialize()


def obj_id_from_url(request):
    # the pyramid URL parser returns a tuple of 0 or more
    # matching items, at least when using the * wild card
    obj_id = request.matchdict.get('obj_id')
    return obj_id[0] if obj_id else None


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
