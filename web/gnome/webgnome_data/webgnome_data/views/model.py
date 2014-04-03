"""
Views for the Model object.
"""
from .common_object import get_object, create_or_update_object

from cornice import Service

model = Service(name='model', path='/model*obj_id', description="Model API")

implemented_types = ('gnome.model.Model',
                     )


@model.get()
def get_model(request):
    '''Returns Model object in JSON.'''
    return get_object(request, implemented_types)


@model.put()
def create_or_update_model(request):
    return create_or_update_object(request, implemented_types)
