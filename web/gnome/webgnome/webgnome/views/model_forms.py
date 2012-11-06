import gnome.model

from pyramid.renderers import render
from pyramid.view import view_config

from webgnome.form_view import FormViewBase
from webgnome.forms import (
    RunModelUntilForm,
    ModelSettingsForm
)
from webgnome import util


class ModelFormViews(FormViewBase):
    """
    Form routes for :class:`gnome.model.Model`.
    """
    wrapped_class = gnome.model.Model

    # Routes
    CREATE_ROUTE = 'create_model'
    UPDATE_ROUTE = 'update_model'
    RUN_UNTIL_ROUTE = 'run_model_until'

    @view_config(route_name=CREATE_ROUTE, renderer='gnome_json')
    def create_model(self):
        """
        Create a new model for the user. Delete the user's current model if one
        exists.
        """
        settings = self.request.registry.settings
        model_id = self.request.session.get(settings.model_session_key, None)
        confirm = self.request.POST.get('confirm_new', None)

        if model_id and confirm:
            settings.Model.delete(model_id)
            model = settings.Model.create()
            model_id = model.id
            self.request.session[settings.model_session_key] = model.id
            message = util.make_message('success', 'Created a new model.')
        else:
            message = util.make_message('error', 'Could not create a new model. '
                                             'Invalid data was received.')

        return {
            'model_id': model_id,
            'message': message
        }

    @view_config(route_name=RUN_UNTIL_ROUTE, renderer='gnome_json')
    @util.json_require_model
    def run_model_until(self, model):
        """
        Render a :class:`webgnome.forms.RunModelUntilForm` for the user's
        current model on GET and validate form input on POST.
        """
        form = RunModelUntilForm(self.request.POST)
        data = {}

        if self.request.method == 'POST' and form.validate():
            date = form.get_datetime()
            model.set_run_until(date)
            return {'run_until': date, 'form_html': None}

        context = {
            'form': form,
            'action_url': self.request.route_url(self.RUN_UNTIL_ROUTE)
        }

        data['form_html'] = render(
            'webgnome:templates/forms/run_model_until.mak', context)

        return data

    @view_config(route_name='model_settings', renderer='gnome_json')
    @util.json_require_model
    def model_settings(self, model):
        form = ModelSettingsForm(self.request.POST)

        if self.request.method == 'POST' and form.validate():
            return {
                'form_html': None
            }

        context = {
            'form': form,
            'action_url': self.request.route_url(self.UPDATE_ROUTE)
        }

        return {
            'form_html': render(
                'webgnome:templates/forms/model_settings.mak', context)
        }
