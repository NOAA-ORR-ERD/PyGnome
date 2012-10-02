from pyramid.view import view_config

@view_config(route_name='index', renderer='templates/index.mak')
def my_view(request):
    return {}

@view_config(route_name='model', renderer='model.mak')
def my_view(request):
    return {}

