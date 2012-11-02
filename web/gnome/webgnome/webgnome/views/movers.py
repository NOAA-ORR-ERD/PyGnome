import numpy

from gnome import simple_mover, movers
from gnome import basic_types

from pyramid.httpexceptions import HTTPNotFound
from pyramid.renderers import render
from pyramid.view import view_config

from webgnome.forms import (
    VariableWindMoverForm,
    ConstantWindMoverForm
)

from webgnome.form_view import FormView
from webgnome.util import json_require_model, make_message


class WindMoverFormView(FormView):
    """
    Form routes for :class:`gnome.movers.WindMover`.
    """
    wrapped_class = movers.WindMover
    ADD_CONSTANT_WIND = 'add_constant_wind_mover'
    EDIT_CONSTANT_WIND = 'edit_constant_wind_mover'
    ADD_VARIABLE_WIND = 'add_variable_wind_mover'
    EDIT_VARIABLE_WIND = 'edit_variable_wind_mover'

    def _get_route_for_object(self, wind_mover):
        """
        Return the correct route for ``obj``.
        :param obj: a :class:`WindMover` instance
        :return: a string route name
        """
        if not wind_mover.id:
            return

        if wind_mover.is_constant:
            return self.EDIT_CONSTANT_WIND
        else:
            return self.EDIT_VARIABLE_WIND

    def _edit_constant_wind_mover_post(self, model, mover_id, form):
        if model.has_mover_with_id(mover_id):
            # TODO: Update the mover with settings in POST.
            message = make_message(
                'success', 'Updated constant wind mover successfully.')
        else:
            mover_id = model.add_mover(form.data)
            message = make_message(
                'warning',
                'The specified mover did not exist. Added a new constant '
                'wind mover to the model.')

        return {
            'id': mover_id,
            'message': message,
            'form_html': None
        }

    @view_config(route_name=EDIT_CONSTANT_WIND, renderer='gnome_json')
    @json_require_model
    def edit_constant_wind_mover(self, model):
        mover_id = self.request.matchdict['id']
        mover = model.get_mover(mover_id)
        # TODO: Use when real mover class is available.
        # opts = {'obj': mover} if mover else {}
        opts = {}
        form = ConstantWindMoverForm(self.request.POST or None, **opts)

        if self.request.method == 'POST' and form.validate():
            return self._edit_constant_wind_mover_post(model, mover_id, form)

        html = render('webgnome:templates/forms/constant_wind_mover.mak', {
            'form': form,
            'action_url': self.request.route_url(
                self.EDIT_CONSTANT_WIND, id=mover_id)
        })

        return {'form_html': html}


    def _edit_variable_wind_mover_post(self, model, mover_id, form):
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

    @view_config(route_name=EDIT_VARIABLE_WIND, renderer='gnome_json')
    @json_require_model
    def edit_variable_wind_mover(self, model):
        mover_id = self.request.matchdict['id']
        mover = model.get_mover(mover_id)
        # TODO: Use when real mover class is available.
        # opts = {'obj': mover} if mover else {}
        opts = {}
        form = VariableWindMoverForm(self.request.POST or None, **opts)

        if self.request.method == 'POST' and form.validate():
            return self._edit_variable_wind_mover_post(model, mover_id, form)

        html = render('webgnome:templates/forms/variable_wind_mover.mak', {
            'form': form,
            'action_url': self.request.route_url(
                self.EDIT_VARIABLE_WIND, id=mover_id)
        })

        return {'form_html': html}

    def _add_constant_wind_mover_post(self, model, form):
        # TODO: Use ConstantWindMover class when ready.
        time_val = numpy.zeros((1,), dtype=basic_types.time_value_pair)
        time_val['time'][0] = 0  # constant time
        time_val['value'][0] = (0., 100.)
        mover = movers.WindMover(wind_vel=time_val)

        return {
            'id': model.add_mover(mover),
            'type': 'mover',
            'form_html': None
        }

    @view_config(route_name=ADD_CONSTANT_WIND, renderer='gnome_json')
    @json_require_model
    def add_constant_wind_mover(self, model):
        form = ConstantWindMoverForm(self.request.POST)

        if self.request.method == 'POST' and form.validate():
            return self._add_constant_wind_mover_post(model, form)

        html = render('webgnome:templates/forms/constant_wind_mover.mak', {
            'form': form,
            'action_url': self.request.route_url(self.ADD_CONSTANT_WIND)
        })

        return {'form_html': html}

    def _add_variable_wind_mover_post(self, model, form):
        # TODO: Use VariableWindMover class when ready.
        mover = simple_mover.SimpleMover(velocity=(1.0, 10.0, 0.0))

        return {
            'id': model.add_mover(mover),
            'type': 'mover',
            'form_html': None
        }

    @view_config(route_name=ADD_VARIABLE_WIND, renderer='gnome_json')
    @json_require_model
    def add_variable_wind_mover(self, model):
        form = VariableWindMoverForm(self.request.POST)

        if self.request.method == 'POST' and form.validate():
            return self._add_variable_wind_mover_post(model, form)

        context = {
            'form': form,
            'action_url': self.request.route_url(self.ADD_VARIABLE_WIND)
        }

        return {
            'form_html': render(
                'webgnome:templates/forms/variable_wind_mover.mak', context)
        }


@view_config(route_name='delete_mover', renderer='gnome_json',
             request_method='POST')
@json_require_model
def delete_mover(request, model):
    mover_id = request.POST.get('mover_id', None)

    if mover_id is None or model.has_mover_with_id(mover_id) is False:
        raise HTTPNotFound

    model.delete_mover(mover_id)

    return {
        'success': True
    }
