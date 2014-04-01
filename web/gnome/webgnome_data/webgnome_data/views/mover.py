"""
Views for the Mover objects.
This currently includes ??? objects.
"""
from .common_object import get_object, create_or_update_object

from cornice import Service

env = Service(name='mover', path='/mover*obj_id',
              description="Mover API")

implemented_types = ('gnome.movers.simple_mover.SimpleMover',
                     'gnome.movers.wind_movers.WindMover',
                     'gnome.movers.wind_movers.GridWindMover',
                     'gnome.movers.random_movers.RandomMover',
                     'gnome.movers.random_movers.RandomVerticalMover',
                     'gnome.movers.current_movers.CatsMover',
                     'gnome.movers.current_movers.ComponentMover',
                     'gnome.movers.current_movers.GridCurrentMover',
                     'gnome.movers.vertical_movers.RiseVelocityMover',
                     )


@env.get()
def get_mover(request):
    '''Returns an Gnome Environment object in JSON.'''
    return get_object(request, implemented_types)


@env.put()
def create_or_update_mover(request):
    '''Creates or Updates an Environment object.'''
    return create_or_update_object(request, implemented_types)
