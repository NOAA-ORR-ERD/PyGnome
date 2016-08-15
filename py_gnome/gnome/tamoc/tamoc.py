#!/usr/bin/env python

"""
Assorted code for working with TAMOC
"""

from datetime import timedelta

import numpy as np

from gnome.utilities import serializable
from gnome.utilities.projections import FlatEarthProjection

from gnome.utilities.distributions import WeibullDistribution

from .. import _valid_units

__all__ = []

# def tamoc_spill(release_time,
#                 start_position,
#                 num_elements=None,
#                 end_release_time=None,
#                 name='TAMOC plume'):
#     '''
#     Helper function returns a Spill object for a spill from the TAMOC model

#     This version is essentially a template -- it needs to be filled in with
#     access to the parameters from the "real" TAMOC model.

#     Also, this version is for inert particles only a size and density.
#     They will not change once released into gnome.

#     Future work: create a "proper" weatherable oil object.

#     :param release_time: start of plume release
#     :type release_time: datetime.datetime

#     :param start_position: location of initial release
#     :type start_position: 3-tuple of floats (long, lat, depth)

#     :param num_elements: total number of elements to be released
#     :type num_elements: integer

#     :param end_release_time=None: End release time for a time varying release.
#                                   If None, then release runs for tehmodel duration
#     :type end_release_time: datetime.datetime

#     :param float flow_rate=None: rate of release mass or volume per time.
#     :param str units=None: must provide units for amount spilled.
#     :param tuple windage_range=(.01, .04): Percentage range for windage.
#                                            Active only for surface particles
#                                            when a mind mover is added
#     :param windage_persist=900: Persistence for windage values in seconds.
#                                 Use -1 for inifinite, otherwise it is
#                                 randomly reset on this time scale.
#     :param str name='TAMOC spill': a name for the spill.
#     '''

#     release = PointLineRelease(release_time=release_time,
#                                start_position=start_position,
#                                num_elements=num_elements,
#                                end_release_time=end_release_time)

#     # This helper function is just passing parameters thru to the plume
#     # helper function which will do the work.
#     # But this way user can just specify all parameters for release and
#     # element_type in one go...
#     element_type = elements.plume(distribution_type=distribution_type,
#                                   distribution=distribution,
#                                   substance_name=substance,
#                                   windage_range=windage_range,
#                                   windage_persist=windage_persist,
#                                   density=density,
#                                   density_units=density_units)

#     return Spill(release,
#                  element_type=element_type,
#                  amount=amount,
#                  units=units,
#                  name=name)


class TamocDroplet():
    """
    Dummy class to show what we need from the TAMOC output
    """
    def __init__(self,
                 mass_flux=1.0,  # kg/s
                 radius=1e-6,  # meters
                 density=900.0,  # kg/m^3 at 15degC
                 position=(10, 20, 100)  # (x, y, z) in meters
                 ):

        self.mass_flux = mass_flux
        self.radius = radius
        self.density = density
        self.position = np.asanyarray(position)


def test_tamoc_results():
    """
    utility for providing a tamoc result set

    a simple list of TamocDroplet objects
    """
    num_droplets = 10

    mass_flux = np.ones((num_droplets,)) * 1.0  # kg/s

    radius = np.linspace(1e-6, 100, num_droplets)
    density = np.ones((num_droplets,)) * 900  # kg/m^3 at 15degC

    # linear release
    position = np.empty((num_droplets, 3), dtype=np.float64)
    position[:, 0] = np.linspace(10, 50, num_droplets)  # x
    position[:, 0] = np.linspace(5, 25, num_droplets)  # y
    position[:, 0] = np.linspace(20, 100, num_droplets)  # z

    results = [TamocDroplet(*params) for params in zip(mass_flux,
                                                       radius,
                                                       density,
                                                       position)]

    return results


class TamocSpill(serializable.Serializable):
    """
    Models a spill
    """
    # _update = ['on', 'release',
    #            'amount', 'units', 'amount_uncertainty_scale']

    # _create = ['frac_coverage']
    # _create.extend(_update)

    # _state = copy.deepcopy(serializable.Serializable._state)
    # _state.add(save=_create, update=_update)
    # _state += serializable.Field('element_type',
    #                              save=True,
    #                              save_reference=True,
    #                              update=True)
    # _schema = SpillSchema

    # valid_vol_units = _valid_units('Volume')
    # valid_mass_units = _valid_units('Mass')

    def __init__(self,
                 release_time,
                 start_position,
                 num_elements=None,
                 end_release_time=None,
                 name='TAMOC plume',
                 TAMOC_interval=None,
                 on=True,
                 ):
        """

        """

        self.release_time = release_time
        self.start_position = start_position
        self.num_elements = num_elements
        self.end_release_time = end_release_time
        self.num_released = 0
        self.amount_released = 0

        self.tamoc_interval = timedelta(hours=TAMOC_interval) if TAMOC_interval is not None else None
        self.last_tamoc_time = release_time
        self.droplets = None
        self.on = on    # spill is active or not
        self.name = name

    def run_tamoc(self, current_time, time_step):
        # runs TAMOC if no droplets have been initialized or if current_time has reached last_tamoc_run + interval
        if self.on:
            if self.tamoc_interval is None:
                if self.last_tamoc_time is None:
                    self.last_tamoc_time = current_time
                    self.droplets = self._run_tamoc()
                return self.droplets

            if (self.current_time > release_time and (last_tamoc_time is None or self.droplets is None) or
                self.current_time > self.last_tamoc_time + self.tamoc_interval and self.current_time < end_release_time):
                self.last_tamoc_time = current_time
                self.droplets =  self._run_tamoc()
        return self.droplets

    def _run_tamoc(self):
        """
        this is the code that actually calls and runs tamoc_output

        it returns a list of TAMOC droplet objects
        """
        return test_tamoc_results()

    def __repr__(self):
        return ('{0.__class__.__module__}.{0.__class__.__name__}()'.format(self))

    def _get_mass_distribution(self, mass_fluxes, time_step):
        ts = time_step.total_seconds()
        delta_masses = []
        for i, flux in enumerate(mass_fluxes):
            delta_masses.append(mass_fluxes * ts)
        total_mass = sum(delta_masses)
        proportions = [d_mass / total_mass for d_mass in delta_masses]

        return (delta_masses, proportions, total_mass)

    @property
    def units(self):
        """
        Default units in which amount of oil spilled was entered by user.
        The 'amount' property is returned in these 'units'
        """
        return self._units

    @units.setter
    def units(self, units):
        """
        set default units in which volume data is returned
        """
        self._check_units(units)  # check validity before setting
        self._units = units

    def _check_units(self, units):
        """
        Checks the user provided units are in list of valid volume
        or mass units
        """

        if (units in self.valid_vol_units or
                units in self.valid_mass_units):
            return True
        else:
            msg = ('Units for amount spilled must be in volume or mass units. '
                   'Valid units for volume: {0}, for mass: {1} ').format(
                       self.valid_vol_units, self.valid_mass_units)
            ex = uc.InvalidUnitError(msg)
            self.logger.exception(ex, exc_info=True)
            raise ex  # this should be raised since run will fail otherwise

    # what is this for??
    def get_mass(self, units=None):
        '''
        Return the mass released during the spill.
        User can also specify desired output units in the function.
        If units are not specified, then return in 'SI' units ('kg')
        If volume is given, then use density to find mass. Density is always
        at 15degC, consistent with API definition
        '''
        # first convert amount to 'kg'
        if self.units in self.valid_mass_units:
            mass = uc.convert('Mass', self.units, 'kg', self.amount_released)

        if units is None or units == 'kg':
            return mass
        else:
            self._check_units(units)
            return uc.convert('Mass', 'kg', units, mass)

    def uncertain_copy(self):
        """
        Returns a deepcopy of this spill for the uncertainty runs

        The copy has everything the same, including the spill_num,
        but it is a new object with a new id.

        Not much to this method, but it could be overridden to do something
        fancier in the future or a subclass.

        There are a number of python objects that cannot be deepcopied.
        - Logger objects

        So we copy them temporarily to local variables before we deepcopy
        our Spill object.
        """
        u_copy = copy.deepcopy(self)
        self.logger.debug(self._pid + "deepcopied spill {0}".format(self.id))

        return u_copy

    def rewind(self):
        """
        rewinds the release to original status (before anything has been
        released).
        """
        self.num_released = 0
        self.amount_released = 0
        self.droplets = self.run_tamoc()


    def num_elements_to_release(self, current_time, time_step):
        """
        Determines the number of elements to be released during:
        current_time + time_step

        It invokes the num_elements_to_release method for the the unerlying
        release object: self.release.num_elements_to_release()

        :param current_time: current time
        :type current_time: datetime.datetime
        :param int time_step: the time step, sometimes used to decide how many
            should get released.

        :returns: the number of elements that will be released. This is taken
            by SpillContainer to initialize all data_arrays.
        """
        if ~self.on:
            return 0

        if current_time < self.release_time or current_time > self.end_release_time:
            return 0

        self.droplets = self.run_tamoc(current_time, time_step)

        duration = (self.end_release_time - self.release_time).total_seconds()
        if duration is 0:
            duration = 1
        LE_release_rate = self.num_elements / duration
        num_to_release = int(LE_release_rate * time_step.total_seconds())
        if num_released + num_to_release > num_elements:
            num_to_release = num_elements - num_released

        return num_to_release

        #return self.release.num_elements_to_release(current_time, time_step)

    def set_newparticle_values(self, num_new_particles, current_time,
                               time_step, data_arrays):
        """
        SpillContainer will release elements and initialize all data_arrays
        to default initial value. The SpillContainer gets passed as input and
        the data_arrays for 'position' get initialized correctly by the release
        object: self.release.set_newparticle_positions()

        If a Spill Amount is given, the Spill object also sets the 'mass' data
        array; else 'mass' array remains '0'

        :param int num_new_particles: number of new particles that were added.
            Always greater than 0
        :param current_time: current time
        :type current_time: datetime.datetime
        :param time_step: the time step, sometimes used to decide how many
            should get released.
        :type time_step: integer seconds
        :param data_arrays: dict of data_arrays provided by the SpillContainer.
            Look for 'positions' array in the dict and update positions for
            latest num_new_particles that are released
        :type data_arrays: dict containing numpy arrays for values

        Also, the set_newparticle_values() method for all element_type gets
        called so each element_type sets the values for its own data correctly
        """
        mass_fluxes = [tam_drop.mass_flux for tam_drop in self.droplets]
        delta_masses, proportions, total_mass = self._get_mass_distribution(mass_fluxes, time_step)

        #set up LE distribution, the number of particles in each 'release point'
        LE_distribution = [int(num_new_particles * p) for p in proportions]
        diff = num_new_particles - sum(LE_distribution)
        for i in range(0, diff):
            LE_distribution[i % len(LE_distribution)] += 1


        #compute release point location for each droplet
        positions = [self.start_position + FlatEarthProjection.meters_to_lonlat(d.position, self.start_position) for d in self.droplets]

        #for each release location, set the position and mass of the elements released at that location
        total_rel = 0
        for mass_dist, n_LEs ,pos in (delta_masses, LE_distribution, positions):
            start_idx = -num_new_particles + total_rel
            end_idx = start_idx + n_LEs

            data_arrays['positions'][start_idx:end_idx] = pos
            data_arrays['mass'][start_idx:end_idx] = mass_dist / n_LEs
            data_arrays['init_mass'][start_idx:end_idx] = mass_dist / n_LEs
            total_rel += n_LEs

        self.num_released += num_new_particles
        self.amount += total_mass

        # if self.element_type is not None:
        #     self.element_type.set_newparticle_values(num_new_particles, self,
        #                                              data_arrays)

        # self.release.set_newparticle_positions(num_new_particles, current_time,
        #                                        time_step, data_arrays)

        # data_arrays['mass'][-num_new_particles:] = \
        #     self._elem_mass(num_new_particles, current_time, time_step)

        # # set arrays that are spill specific - 'frac_coverage'
        # if 'frac_coverage' in data_arrays:
        #     data_arrays['frac_coverage'][-num_new_particles:] = \
        #         self.frac_coverage
