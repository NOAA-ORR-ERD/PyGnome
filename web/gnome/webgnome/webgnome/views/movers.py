import numpy

from gnome import simple_mover, movers

from pyramid.httpexceptions import HTTPFound, HTTPNotFound
from pyramid.renderers import render
from pyramid.view import view_config

from gnome import basic_types

from ..forms import (
    VariableWindMoverForm,
    ConstantWindMoverForm
)

from ..util import json_require_model, make_message


@view_config(route_name='edit_constant_wind_mover', renderer='gnome_json')
@json_require_model
def edit_constant_wind_mover(request, model):
    mover_id = request.matchdict['id']
    mover = model.get_mover(mover_id)
    # TODO: Use when real mover class is available.
    # opts = {'obj': mover} if mover else {}
    opts = {}
    form = ConstantWindMoverForm(request.POST or None, **opts)

    if request.method == 'POST' and form.validate():
        if model.has_mover_with_id(mover_id):
            # TODO: Update the mover with settings in POST.
            message = make_message(
                'success', 'Updated constant wind mover successfully.')
        else:
            mover_id = model.add_mover(form.data)
            message = make_message('warning',
                                    'The specified mover did not exist. Added '
                                    'a new constant wind mover to the model.')
        return {
            'id': mover_id,
            'message': message,
            'form_html': None
        }

    html = render('webgnome:templates/forms/constant_wind_mover.mak', {
        'form': form,
        'action_url': request.route_url(
            'edit_constant_wind_mover', id=mover_id)
    })

    return {'form_html': html}


@view_config(route_name='edit_variable_wind_mover', renderer='gnome_json')
@json_require_model
def edit_variable_wind_mover(request, model):
    mover_id = request.matchdict['id']
    mover = model.get_mover(mover_id)
    # TODO: Use when real mover class is available.
    # opts = {'obj': mover} if mover else {}
    opts = {}
    form = VariableWindMoverForm(request.POST or None, **opts)

    if request.method == 'POST' and form.validate():
        if model.has_mover_with_id(mover_id):
            # TODO: Update the mover with settings in POST.
            message = make_message(
                'success', 'Updated variable wind mover successfully.')
        else:
            mover_id = model.add_mover(form.data)
            message = make_message('warning',
                                    'The specified mover did not exist. Added '
                                    'a new variable wind mover to the model.')
        return {
            'id': mover_id,
            'message': message,
            'form_html': None
        }

    html = render('webgnome:templates/forms/variable_wind_mover.mak', {
        'form': form,
        'action_url': request.route_url('edit_variable_wind_mover',
                                        id=mover_id)
    })

    return {'form_html': html}


@view_config(route_name='add_constant_wind_mover', renderer='gnome_json')
@json_require_model
def add_constant_wind_mover(request, model):
    form = ConstantWindMoverForm(request.POST)

    if request.method == 'POST' and form.validate():
        # TODO: Use ConstantWindMover class when ready.
        time_val = numpy.zeros((1,), dtype=basic_types.time_value_pair)
        time_val['time'][0] = 0  # since it is just constant, just give it 0 time
        time_val['value'][0] = (0., 100.)
        mover = movers.WindMover(wind_vel=time_val)

        return {
            'id': model.add_mover(mover),
            'type': 'mover',
            'form_html': None
        }

    html = render('webgnome:templates/forms/constant_wind_mover.mak', {
        'form': form,
        'action_url': request.route_url('add_constant_wind_mover')
    })

    return {'form_html': html}


@view_config(route_name='add_variable_wind_mover', renderer='gnome_json')
@json_require_model
def add_variable_wind_mover(request, model):
    form = VariableWindMoverForm(request.POST)

    if request.method == 'POST' and form.validate():
        # TODO: Use VariableWindMover class when ready.
        
        mover = simple_mover.SimpleMover(velocity=(1.0, 10.0, 0.0))

        return {
            'id': model.add_mover(mover),
            'type': 'mover',
            'form_html': None
        }

    context = {
        'form': form,
        'action_url': request.route_url('add_variable_wind_mover')
    }

    return {
        'form_html': render(
            'webgnome:templates/forms/variable_wind_mover.mak', context)
    }


@view_config(route_name='delete_mover', renderer='gnome_json', request_method='POST')
@json_require_model
def delete_mover(request, model):
    mover_id = request.POST.get('mover_id', None)

    if mover_id is None or model.has_mover_with_id(mover_id) is False:
        raise HTTPNotFound

    model.delete_mover(mover_id)

    return {
        'success': True
    }
