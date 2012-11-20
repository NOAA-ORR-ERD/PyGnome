import numpy
import gnome.movers
import gnome.basic_types

from pyramid.renderers import render
from pyramid.view import view_config

from webgnome import util
from webgnome.forms.movers import AddMoverForm, DeleteMoverForm, WindMoverForm


# A map of :mod:`gnome` objects to route names, for use looking up the route
# for an object at runtime with :func:`get_form_route`.
form_routes = {
    gnome.movers.WindMover: {
        'create': 'create_wind_mover',
        'update': 'update_wind_mover',
        'delete': 'delete_mover'
    },
}


def get_form_route(obj, route_type):
    """
    Find a route name for ``obj`` given the type of route.

    ``route_type`` is a short-hand description like "create" or "delete" used
    as a key in the ``form_routes`` dictionary.
    """
    route = None
    form_cls = util.get_obj_class(obj)
    routes = form_routes.get(form_cls, None)

    if routes:
        route = routes.get(route_type, None)

    return route


@view_config(route_name='create_mover', renderer='gnome_json')
@util.json_require_model
def create_mover(request, model):
    form = AddMoverForm(request.POST)

    if request.method == 'POST' and form.validate():
        # TODO: What does the client need to know at this point?
        return {
            'success': True
        }

    context = {
        'form': form,
        'action_url': request.route_url('create_mover')
    }

    return {
        'form_html': render(
            'webgnome:templates/forms/add_mover.mak', context)
    }


@view_config(route_name='delete_mover', renderer='gnome_json',
             request_method='POST')
@util.json_require_model
def delete_mover(request, model):
    form = DeleteMoverForm(request.POST, model)

    if form.validate():
        mover_id = form.mover_id
        model.delete_mover(mover_id)

        return {
            'success': True
        }

    context = {
        'form': form,
        'action_url': request.route_url('delete_mover'),
    }

    return {
        'form_html': render(
            'webgnome:templates/forms/delete_mover.mak', context)
    }


def _update_wind_mover_post(model, mover_id, form):
    if model.has_mover_with_id(mover_id):
        # TODO: Does WTForms update the object directly?
        message = util.make_message(
            'success', 'Updated variable wind mover successfully.')
    else:
        mover_id = model.add_mover(form.data)
        message = util.make_message(
            'warning', 'The specified mover did not exist. Added a '
                       'new variable wind mover to the model.')
    return {
        'id': mover_id,
        'message': message,
        'form_html': None
    }


@view_config(route_name='update_wind_mover', renderer='gnome_json')
@util.json_require_model
def update_wind_mover(request, model):
    mover_id = request.matchdict['id']
    mover = model.get_mover(mover_id)
    opts = {'obj': mover} if mover else {}
    form = WindMoverForm( request.POST or None, **opts)

    if request.method == 'POST' and form.validate():
        return _update_wind_mover_post(model, mover_id, form)

    html = render('webgnome:templates/forms/wind_mover.mak', {
        'form': form,
        'action_url': request.route_url('update_wind_mover', id=mover_id)
    })

    return {'form_html': html}


def _create_wind_mover_post(model, form):
    time_series = numpy.zeros((1,), dtype=gnome.basic_types.datetime_value_2d)

    for time_form in form.time_series:
        direction = time_form.get_direction_degree()

        if not direction:
            return {
                'form_html': None,
                'message': util.make_message('error',
                    'Could not create wind mover. Invalid direction given.')
            }

        time_series['time'][0] = 0
        time_series['value'][0] = (direction, time_form.speed.data)

    mover = gnome.movers.WindMover(timeseries=time_series)

    return {
        'id': model.add_mover(mover),
        'type': 'mover',
        'form_html': None
    }


@view_config(route_name='create_wind_mover', renderer='gnome_json')
@util.json_require_model
def create_wind_mover(request, model):
    form = WindMoverForm(request.POST)

    if request.method == 'POST' and form.validate():
        return _create_wind_mover_post(model, form)

    context = {
        'form': form,
        'action_url': request.route_url('create_wind_mover')
    }

    return {
        'form_html': render(
            'webgnome:templates/forms/wind_mover.mak', context)
    }

