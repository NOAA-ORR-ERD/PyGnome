#!/usr/bin/env python

"""
assorted code for working with TAMOC
"""

__all__ = []

def tamoc_spill(num_elements,
                position,
                release_time,
                windage_range=(.01, .04),
                windage_persist=900,
                name='TAMOC plume'):
    '''
    Helper function returns a Spill object for a spill from the TAMOC model

    This version is essentially a template -- it needs to be filled in with 
    access to the parameters from the "real" TAMOC model.

    Also, this version is for intert particles -- they will not change once released into gnome.

    Future work: create a "proper" weatherable oil object.

    :param num_elements: total number of elements to be released
    :type num_elements: integer

    :param position: location of initial release
    :type start_position: 3-tuple of floats (long, lat, depth)

    :param release_time: start of plume release
    :type release_time: datetime.datetime

    :param end_release_time=None: End release time for a time varying release.
                                  If None, then release runs for tehmodel duration
    :type end_release_time: datetime.datetime

    :param float flow_rate=None: rate of release mass or volume per time.
    :param str units=None: must provide units for amount spilled.
    :param tuple windage_range=(.01, .04): Percentage range for windage.
                                           Active only for surface particles
                                           when a mind mover is added
    :param windage_persist=900: Persistence for windage values in seconds.
                                Use -1 for inifinite, otherwise it is
                                randomly reset on this time scale.
    :param str name='TAMOC spill': a name for the spill.
    '''

    release = PointLineRelease(release_time=release_time,
                               start_position=start_position,
                               num_elements=num_elements,
                               end_release_time=end_release_time)

    # This helper function is just passing parameters thru to the plume
    # helper function which will do the work.
    # But this way user can just specify all parameters for release and
    # element_type in one go...
    element_type = elements.plume(distribution_type=distribution_type,
                                  distribution=distribution,
                                  substance_name=substance,
                                  windage_range=windage_range,
                                  windage_persist=windage_persist,
                                  density=density,
                                  density_units=density_units)

    return Spill(release,
                 element_type=element_type,
                 amount=amount,
                 units=units,
                 name=name)
