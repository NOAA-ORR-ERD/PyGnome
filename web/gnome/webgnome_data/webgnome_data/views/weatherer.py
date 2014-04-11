"""
Views for the Weatherer objects.
"""
from .common_object import get_object, create_or_update_object

from cornice import Service

env = Service(name='weatherer', path='/weatherer*obj_id',
              description="Weatherer API")

implemented_types = ('gnome.weatherers.core.Weatherer',
                     )


@env.get()
def get_weatherer(request):
    '''Returns a Gnome Weatherer object in JSON.'''
    return get_object(request, implemented_types)


@env.put()
def create_or_update_weatherer(request):
    '''Creates or Updates a Weatherer object.'''
    return create_or_update_object(request, implemented_types)
