
# don't have tamoc running right now.
# from gnome_objs import TamocBlowout

from datetime import datetime, timedelta
import numpy as np

import unit_conversion as uc

from gnome.utilities.time_utils import asdatetime
from gnome.spill.spill import BaseSpill
from gnome.spill.release import PointLineRelease

from gnome.array_types import gat


class DummyTamocBlowout(object):
    """
    Dummy TamocBlowout Object to use for testing

    The real one is defined in gnome_objs
    """
    def __init__(self,
                 z0,
                 d0,
                 adios_id,
                 q_oil,
                 gor,
                 x0=0.,
                 y0=0.,
                 u0=0.,
                 phi_0=-np.pi / 2.,
                 theta_0 = 0.,
                 ):
        """
        Initializer for a tamoc_blowout object.
        """
        super(TamocBlowout, self).__init__()

        # Store required class attributes passed to initializer
        self.z0 = z0
        self.d0 = d0
        self.adios_id = adios_id
        self.q_oil = q_oil
        self.gor = gor

        # Store optional class attributes
        self.x0 = x0
        self.y0 = y0
        self.u0 = u0
        self.phi_0 = phi_0
        self.theta_0 = theta_0
        self.ca = ['nitrogen', 'oxygen', 'argon', 'carbon_dioxide']

    def simulate(self):
        """
        this would be when it runs
        """
        pass


TamocBlowout = DummyTamocBlowout


class TamocSpill(BaseSpill):
    """
    GNOME Spill object that uses TAMOC to initialize the data

    """

    def __init__(self,
                 num_elements=1000,
                 start_position=(0.0, 0.0, 100.),
                 release_time=datetime.now(),
                 release_duration=timedelta(hours=12),
                 tamoc_time_delta=timedelta(hours=1),
                 substance='AD01554',
                 release_rate=20000.,
                 release_units='bbl/day',
                 gor=0.,
                 d0=0.1,
                 phi_0=-np.pi / 2.,
                 theta_0=0.,
                 current=None,
                 water=None,
                 on=True,
                 name='My blowout',
                 **kwargs):


        # Compute the total amount spilled from the release rate and duration
        duration = release_duration.total_seconds()
        try:
            amount = uc.convert(release_units, 'm^3/s', release_rate) * \
                        duration
            units = 'm^3'
        except uc.NotSupportedUnitError:
            amount = uc.convert(release_units, 'kg/s', release_rate) * \
                        duration
            units = 'kg'

        super(TamocSpill, self).__init__(num_elements=num_elements,
                                         amount=amount,
                                         units=units,
                                         substance=substance,
                                         on=on,
                                         name=name,
                                         **kwargs)

        # Initialize object attributes unique to TamocBlowout
        self.array_types = {'droplet_diameter': gat('droplet_diameter')}
        self.release_time = asdatetime(release_time)
        self.end_release_time = asdatetime(release_time) + release_duration
        self.release_rate = release_rate
        self.release_units = release_units
        self.water = water
        self.tamoc_time_delta = tamoc_time_delta

        # Compute some of the attributes related to the number of elements
        self.num_elements = num_elements
        total_time = (self.end_release_time -
                      self.release_time).total_seconds()
        self.num_elements_per_second = float(self.num_elements) / total_time

        # Initialize a TAMOC model simulation object
        y0, x0, z0 = (start_position[i] for i in range(3))
        # fixit: need to convert these lat/lon values to meters...using a
        # generic conversion that will be wrong
        x0 = x0 * 111319.458
        y0 = y0 * 110574.2727
        # fixit: TAMOC is not working with three-dimensional currents...
        # figure out how to accept u0 = current.velocity
        if current is None:
            u0 = 0.1
        else:
            # fixit: check which component of velocity is in the longitude
            # direction.
            u0 = current.velocity[0]
        self.tamoc_sim = TamocBlowout(z0,
                                           d0,
                                           substance,
                                           release_rate,
                                           gor,
                                           x0,
                                           y0,
                                           u0,
                                           phi_0,
                                           theta_0)

    def prepare_for_model_run(self, timestep):
        """
        Do anything that needs to happen before the first time-step

        """
        pass

    def release_elements(self, sc, current_time, time_step):
        """
        Releases and partially initializes new LEs

        """
        if not self.on:
            return 0

        # Check whether TAMOC needs to be run
        elapsed_time = (current_time - self.release_time).total_seconds()
        # this could be tricky if the numbers don't work out exactly right
        # could probably do this with timedelta math
        if current_time < self.end_release_time and \
            elapsed_time % self.tamoc_time_delta.total_seconds() == 0:
            # Run tamoc
            self.run_tamoc(current_time, time_step)

        # Release the LEs needed for this time step.


        # Compute the expected number of elements to release
        expected_num_release = self.num_elements_after_time(current_time,
                                                            time_step)

        print "   Plan to have released:",  expected_num_release
        actual_num_release = self._num_released
        to_rel = expected_num_release - actual_num_release
        if to_rel <= 0:
            return 0 #nothing to release, so end early

        print '   --> Will release ' + str(to_rel) + ' this time step.'

        # Add elements to the spill container
        sc._append_data_arrays(to_rel)
        self._num_released += to_rel

        # Get the dictionary keyword for the present spill
        # you should not need this!
        idx = sc.spills.index(self)
        sc['spill_num'][-to_rel:] = idx # there should be a better way to do this.

        #if 'frac_coverage' in sc:
        #    sc['frac_coverage'][-to_rel:] = self.frac_coverage

        self.substance.initialize_LEs(to_rel, sc)

        # Update the LE properties with output from TAMOC
        self.initialize_LEs(to_rel, sc, current_time, time_step)

        return to_rel

    def run_tamoc(self, current_time, time_step):
        """
        Run a tamoc simulation for the present conditions

        """
        # Simulate TAMOC
        self.update_tamoc_parameters()
        self.tamoc_sim.simulate()

    def update_tamoc_parameters(self):
        """
        Update the environment forcing the TAMOC simulation

        """
        # fixit: need to add the functionality here.
        return 0

    def num_elements_after_time(self, current_time, time_step):
        """
        Compute the number of elements that should exist for this spill
        after current_time + time_step

        """
        if current_time > self.end_release_time:
            # If release has ended, all elements will exist
            num_released = self.num_elements
        else:
            # Compute total number released from release rate
            num_released = int(((current_time -
                                 self.release_time).total_seconds() +
                                 time_step) * self.num_elements_per_second)

        # Return the total
        return num_released

    def initialize_LEs(self, to_rel, sc, current_time, time_step):
        """
        Initialize the mass and position of each new LE to be release.  This
        follows the actions in gnome.spill.release.initialize_LEs.

        Note:  This function uses the fact that Python lists are pointers to
        memory locations.  When values in the spill container lists are
        updated in this functions, those values are updated in the spill
        container throughout the model.

        """
        # Get the indicies in the spill container that need to be updated
        sl = slice(-to_rel, None, 1)

        # Add a dummy start position for these elements that moves
        # around during the spill so that we can verify that things are
        # changing as needed.
        # fixit: Once working, position should be updated with the latest
        # output from a TAMOC simulation.
        elapsed_time = (current_time - self.release_time).total_seconds()
        total_time = (self.end_release_time -
                      self.release_time).total_seconds()
        if elapsed_time < total_time / 2.:
            position = (0.1, -0.1, 850)
        else:
            position = (0.1, 0.1, 850)

        # Update the position data
        sc['positions'][sl, :] = position

        # Update the masses of each element
        # fixit: use TAMOC to get the correct values of the masses per le
        sc['mass'][sl] = 0.1
        sc['init_mass'][sl] = 0.1
        sc['droplet_diameter'][sl] = 0.001  # in meters

        # fixit: other things to update include:
        print sc['init_mass'][sl]
        print sc['mass'][sl]
        print sc['mass_components'][sl]
        print sc['density'][sl]
        print sc['viscosity'][sl]
        print sc['surface_concentration'][sl]
        print sc['positions'][sl]



