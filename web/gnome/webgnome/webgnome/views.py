from functools import wraps
from pyramid.httpexceptions import HTTPFound
from pyramid.renderers import render
from pyramid.view import view_config

from mock_model import ModelManager
from forms import AddMoverForm, VariableWindMoverForm, ConstantWindMoverForm


MODEL_ID_SESSION_KEY = 'model_id'
MISSING_MODEL_ERROR = 'The model you were working on is no longer available. ' \
                      'We created a new one for you.'


_running_models = ModelManager()


def json_require_model(f):
    """
    Wrap a JSON view in a precondition that checks if the user has a valid
    `model_id` in his or her session and fails if not.

    If the key is missing or no model is found for that key, return a JSON
    object with a `message` object describing the error.
    """
    @wraps(f)
    def inner(request, *args, **kwargs):
        model_id = request.session.get(MODEL_ID_SESSION_KEY, None)
        model = _running_models.get(model_id)

        if model is None:
            return {
                'error': True,
                'message': {
                    'type': 'error',
                    'text': 'That model is no longer available.'
                }
            }
        return f(request, model, *args, **kwargs)
    return inner


@view_config(route_name='show_model', renderer='model.mak')
def show_model(request):
    """
    Show the current user's model.

    Get or create an existing `py_gnome.model.Model` using the `model_id`
    field in the user's session.

    If `model_id` was found in the user's session but the model did not exist,
    warn the user and suggest that they reload from a save file.
    """
    model_id = request.session.get(MODEL_ID_SESSION_KEY, None)
    model, created = _running_models.get_or_create(model_id)
    data = {}

    if created:
        request.session[MODEL_ID_SESSION_KEY] = model.id

        # A model with ID `model_id` did not exist, so we created a new one.
        if model_id:
            data['warning'] = MISSING_MODEL_ERROR

    data['model'] = model

    return data


@view_config(route_name='run_model', renderer='gnome_json')
@json_require_model
def run_model(request, model):
    """
    Run the user's current model and return a JSON object containing the result
    of the run.
    """
    return {
        'result': model.run()
    }


@view_config(route_name='edit_constant_wind_mover', renderer='gnome_json')
@json_require_model
def edit_constant_wind_mover(request, model):
    mover_id = request.matchdict['id']
    mover = model.get_mover(mover_id)
    opts = {'obj': mover} if mover else {}
    form = ConstantWindMoverForm(request.POST or None, **opts)

    if request.method == 'POST' and form.validate():
        if model.has_mover_with_id(mover_id):
            model.update_mover(mover_id, form.data)
            message = {
                'type': 'success',
                'text': 'Updated constant wind mover successfully.'
            }
        else:
            mover_id = model.add_mover(form.data)
            message = {
                 'message': {
                    'type': 'warning',
                    'text': 'The specified mover did not exist. Added a '
                            'new constant wind mover to the model.'
                }
            }

        return {
            'id': mover_id,
            'type': 'mover',
            'message': message
        }

    return {
        'form_html': render(
            'webgnome:templates/forms/constant_wind_mover.mak', {
                'form': form,
                'action_url': request.route_url('edit_constant_wind_mover',
                                                id=mover_id)
            })
    }

@view_config(route_name='add_constant_wind_mover', renderer='gnome_json')
@json_require_model
def add_constant_wind_mover(request, model):
    form = ConstantWindMoverForm(request.POST)

    if request.method == 'POST' and form.validate():
        return {
            'id': model.add_mover(form.data),
            'type': 'mover',
            'message': {
                'type': 'success',
                'text': 'Added a variable wind mover to the model.'
            }
        }

    return {
        'form_html': render(
            'webgnome:templates/forms/constant_wind_mover.mak', {
                'form': form,
                'action_url': request.route_url('add_constant_wind_mover')
            })
    }


@view_config(route_name='add_variable_wind_mover', renderer='gnome_json')
@json_require_model
def add_variable_wind_mover(request, model):
    form = VariableWindMoverForm(request.POST)

    if request.method == 'POST' and form.validate():
        return {
            'id': model.add_mover(form.data),
            'type': 'mover',
            'message': {
                'type': 'success',
                'text': 'Added a variable wind mover to the model.'
            }
        }

    return {
        'form_html': render(
            'webgnome:templates/forms/variable_wind_mover.mak', {
                'form': form,
                'action_url': request.route_url('add_variable_wind_mover')
            })
    }


@view_config(route_name='add_mover', renderer='gnome_json')
@json_require_model
def add_mover(request, model, type=None):
    form = AddMoverForm(request.POST)
    data = {}

    mover_routes = {
        AddMoverForm.MOVER_VARIABLE_WIND: 'add_variable_wind_mover',
        AddMoverForm.MOVER_CONSTANT_WIND: 'add_constant_wind_mover'
    }

    if request.method == 'POST' and form.validate():
        route = mover_routes.get(form.mover_type.data)
        return HTTPFound(request.route_url(route))

    data['form_html'] = render(
        'webgnome:templates/forms/add_mover_form.mak', {
            'form': form, 'action_url': request.route_url('add_mover')
        })

    return data

