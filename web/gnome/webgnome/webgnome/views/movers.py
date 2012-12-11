from pyramid.renderers import render
from pyramid.view import view_config

from webgnome import util
from webgnome.forms.movers import AddMoverForm, DeleteMoverForm, WindMoverForm
from webgnome.model_manager import WebWindMover


# A map of :mod:`gnome` objects to route names, for use looking up the route
# for an object at runtime with :func:`get_form_route`.
form_routes = {
    WebWindMover: {
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
    form = DeleteMoverForm(request.POST, model=model)

    if form.validate():
        model.remove_mover(form.mover_id.data)

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



def _render_wind_mover_form(request, form, mover):
    html = render('webgnome:templates/forms/wind_mover.mak', {
        'form': form,
        'action_url': request.route_url('update_wind_mover', id=mover.id)
    })

    return {'form_html': html}


def _update_wind_mover_post(request, model, mover):
    form = WindMoverForm(request.POST)

    if form.validate():
        if mover:
            form.update(mover)
            message = util.make_message(
                'success', 'Updated variable wind mover successfully.')
        else:
            mover = form.create()
            model.add_mover(mover)
            message = util.make_message(
                'warning', 'The mover did not exist, so we created a new one.')

        return {
            'id': mover.id,
            'message': message,
            'form_html': None
        }

    form.timeseries.append_entry()

    return _render_wind_mover_form(request, form, mover)



@view_config(route_name='update_wind_mover', renderer='gnome_json')
@util.json_require_model
def update_wind_mover(request, model):
    mover_id = request.matchdict['id']
    mover = model.get_mover(int(mover_id))

    if request.method == 'POST':
        return _update_wind_mover_post(request, model, mover)

    form = WindMoverForm(obj=mover)

    return _render_wind_mover_form(request, form, mover)



def _create_wind_mover_post(model, form):
    mover = form.create()

    return {
        'id': model.add_mover(mover),
        'type': 'mover',
        'form_html': None
    }


@view_config(route_name='create_wind_mover', renderer='gnome_json')
@util.json_require_model
def create_wind_mover(request, model):
    form = WindMoverForm(request.POST)

    if request.method == 'POST':
        if form.validate():
            return _create_wind_mover_post(model, form)
        else:
            form.timeseries.append_entry()

    context = {
        'form': form,
        'action_url': request.route_url('create_wind_mover')
    }

    return {
        'form_html': render(
            'webgnome:templates/forms/wind_mover.mak', context)
    }

