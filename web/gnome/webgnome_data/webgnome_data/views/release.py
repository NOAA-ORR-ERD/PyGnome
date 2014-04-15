"""
Views for the Release objects.
"""
from .common_object import get_object, create_or_update_object

from cornice import Service

release = Service(name='release', path='/release*obj_id',
              description="Release API")

implemented_types = ('gnome.spill.release.Release',
                     'gnome.spill.release.PointLineRelease',
                     'gnome.spill.release.SpatialRelease',
                     'gnome.spill.release.VerticalPlumeRelease',
                     )


@release.get()
def get_environment(request):
    '''Returns a Gnome Release object in JSON.'''
    return get_object(request, implemented_types)


@release.put()
def create_or_update_environment(request):
    '''Creates or Updates a Gnome Release object.'''
    return create_or_update_object(request, implemented_types)
