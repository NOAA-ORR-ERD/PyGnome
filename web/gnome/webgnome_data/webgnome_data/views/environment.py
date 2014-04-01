"""
Views for the Environment objects.
This currently includes Wind and Tide objects.
"""
from .common_object import get_object, create_or_update_object

from cornice import Service

env = Service(name='environment', path='/environment*obj_id',
              description="Environment API")

implemented_types = ('gnome.environment.Tide',
                     'gnome.environment.Wind',
                     )


@env.get()
def get_environment(request):
    '''Returns an Gnome Environment object in JSON.'''
    return get_object(request, implemented_types)


@env.put()
def create_or_update_environment(request):
    '''Creates or Updates an Environment object.'''
    return create_or_update_object(request, implemented_types)
