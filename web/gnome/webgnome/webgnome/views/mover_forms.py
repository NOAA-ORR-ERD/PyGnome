import numpy
import gnome.movers
import gnome.basic_types

from pyramid.httpexceptions import HTTPNotFound
from pyramid.renderers import render
from pyramid.view import view_config

from webgnome.forms import AddMoverForm, DeleteMoverForm, WindMoverForm
from webgnome.form_view import FormViewBase
from webgnome.util import json_require_model, make_message



class MoverFormViews(FormViewBase):
    """
    Form views for working with movers of any type: add (i.e., choose mover
    type to add), delete.
    """
    wrapped_class = gnome.movers.PyMover

    # Routes
    CREATE_ROUTE = 'add_mover'
    DELETE_ROUTE = 'delete_mover'

    forms = {
        CREATE_ROUTE: AddMoverForm,
        DELETE_ROUTE: DeleteMoverForm
    }

    @view_config(route_name=CREATE_ROUTE, renderer='gnome_json')
    @json_require_model
    def add_mover(self, request, model):
        form = AddMoverForm(self.request.POST)

        if self.request.method == 'POST' and form.validate():
            # TODO: What does the client need to know at this point?
            return {
                'success': True
            }

        context = {
            'form': form,
            'action_url': self.request.route_url(self.CREATE_ROUTE)
        }

        return {
            'form_html': render(
                'webgnome:templates/forms/add_mover.mak', context)
        }

    @view_config(route_name=DELETE_ROUTE, renderer='gnome_json',
                 request_method='POST')
    @json_require_model
    def delete_mover(self, request, model):
        form = DeleteMoverForm(request.POST, model)

        if form.validate():
            mover_id = form.mover_id
            model.delete_mover(mover_id)

            return {
                'success': True
            }

        context = {
            'form': form,
            'action_url': self.request.route_url(self.DELETE_ROUTE),
        }

        return {
            'form_html': render(
                'webgnome:templates/forms/delete_mover.mak', context)
        }


class WindMoverFormViews(FormViewBase):
    """
    Form routes for :class:`gnome.movers.WindMover`.
    """
    wrapped_class = gnome.movers.WindMover

    # Routes
    CREATE_ROUTE = 'create_wind_mover'
    UPDATE_ROUTE = 'update_wind_mover'

    forms = {
        CREATE_ROUTE: WindMoverForm,
        UPDATE_ROUTE: WindMoverForm
    }

    def _edit_wind_mover_post(self, model, mover_id, form):
        if model.has_mover_with_id(mover_id):
            # TODO: Update the mover with settings in POST.
            message = make_message(
                'success', 'Updated variable wind mover successfully.')
        else:
            mover_id = model.add_mover(form.data)
            message = make_message(
                'warning', 'The specified mover did not exist. Added a '
                           'new variable wind mover to the model.')
        return {
            'id': mover_id,
            'message': message,
            'form_html': None
        }

    @view_config(route_name=UPDATE_ROUTE, renderer='gnome_json')
    @json_require_model
    def edit_wind_mover(self, model):
        mover_id = self.request.matchdict['id']
        # TODO: Use when real mover class is available.
        # opts = {'obj': mover} if mover else {}
        opts = {}
        form = WindMoverForm(self.request.POST or None, **opts)

        if self.request.method == 'POST' and form.validate():
            return self._edit_wind_mover_post(model, mover_id, form)

        html = render('webgnome:templates/forms/wind_mover.mak', {
            'form': form,
            'action_url': self.request.route_url(
                self.UPDATE_ROUTE, id=mover_id)
        })

        return {'form_html': html}

    def _add_wind_mover_post(self, model, form):
        # TODO: Validate form input.
        time_val = numpy.zeros((1,), dtype=gnome.basic_types.time_value_pair)

        # Since it is just constant, just give it 0 time
        time_val['time'][0] = 0
        time_val['value'][0] = (0., 100.)
        mover = gnome.movers.WindMover(wind_vel=time_val)

        return {
            'id': model.add_mover(mover),
            'type': 'mover',
            'form_html': None
        }

    @view_config(route_name=CREATE_ROUTE, renderer='gnome_json')
    @json_require_model
    def add_wind_mover(self, model):
        form = WindMoverForm(self.request.POST)

        if self.request.method == 'POST' and form.validate():
            return self._add_wind_mover_post(model, form)

        context = {
            'form': form,
            'action_url': self.request.route_url(self.CREATE_ROUTE)
        }

        return {
            'form_html': render(
                'webgnome:templates/forms/wind_mover.mak', context)
        }

