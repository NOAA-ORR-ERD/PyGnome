from pyramid.renderers import render
from pyramid.view import view_config


from webgnome.model_manager import WebPointReleaseSpill


# A map of :mod:`gnome` objects to route names, for use looking up the route
# for an object at runtime with :func:`get_form_route`.
form_routes = {
    WebPointReleaseSpill: {
        'create': 'create_wind_mover',
        'update': 'update_wind_mover',
        'delete': 'delete_mover'
    },
}

