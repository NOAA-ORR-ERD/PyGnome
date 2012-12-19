from pyramid.renderers import render
from pyramid.view import view_config

from webgnome import util
from webgnome.forms.spills import (
    AddSpillForm,
    DeleteSpillForm,
    PointReleaseSpillForm
)


@view_config(route_name='create_spill', renderer='gnome_json')
@util.json_require_model
def create_spill(request, model):
    form = AddSpillForm(request.POST)

    if request.method == 'POST' and form.validate():
        return {
            'success': True
        }

    context = {
        'form': form,
        'action_url': request.route_url('create_spill')
    }

    return {
        'form_html': render(
            'webgnome:templates/forms/add_spill.mak', context)
    }


@view_config(route_name='delete_spill', renderer='gnome_json',
             request_method='POST')
@util.json_require_model
def delete_spill(request, model):
    form = DeleteSpillForm(request.POST, model=model)

    if form.validate():
        model.remove_spill(form.obj_id.data)

        return {
            'success': True
        }

    context = {
        'form': form,
        'action_url': request.route_url('delete_spill'),
    }

    return {
        'form_html': render(
            'webgnome:templates/forms/delete_spill.mak', context)
    }


def _render_point_release_spill_form(request, form, spill):
    html = render('webgnome:templates/forms/point_release_spill.mak', {
        'form': form,
        'action_url': request.route_url('update_point_release_spill', id=spill.id)
    })

    return {'form_html': html}


def _update_point_release_spill_post(request, model, spill_id):
    form = PointReleaseSpillForm(request.POST)

    if form.validate():
        spill = form.create()
        model.add_spill(spill)

        if spill:
            model.remove_spill(spill_id)
            message = util.make_message(
                'success', 'Updated point release spill successfully.')
        else:
            message = util.make_message(
                'warning', 'The spill did not exist, so we created a new one.')

        return {
            'id': spill.id,
            'message': message,
            'form_html': None
        }

    return _render_point_release_spill_form(request, form, spill)



@view_config(route_name='update_point_release_spill', renderer='gnome_json')
@util.json_require_model
def update_point_release_spill(request, model):
    spill_id = request.matchdict['id']
    spill = model.get_spill(int(spill_id))

    if request.method == 'POST':
        return _update_point_release_spill_post(request, model, spill.id)

    form = PointReleaseSpillForm(obj=spill)

    return _render_point_release_spill_form(request, form, spill)



def _create_point_release_spill_post(model, form):
    spill = form.create()
    model.add_spill(spill)

    return {
        'id': spill.id,
        'type': 'spill',
        'form_html': None
    }


@view_config(route_name='create_point_release_spill', renderer='gnome_json')
@util.json_require_model
def create_point_release_spill(request, model):
    form = PointReleaseSpillForm(request.POST)

    if request.method == 'POST' and form.validate():
        return _create_point_release_spill_post(model, form)

    context = {
        'form': form,
        'action_url': request.route_url('create_point_release_spill')
    }

    return {
        'form_html': render(
            'webgnome:templates/forms/point_release_spill.mak', context)
    }
