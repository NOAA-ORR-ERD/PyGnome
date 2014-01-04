#!/usr/bin/env python

from datetime import datetime, timedelta
import numpy as np

from gnome.basic_types import world_point


class Plume(object):
    '''
      OK, there is probably a more precise way to do this,
      but I need to keep moving on this.
      Basically, I just need to represent a vertical stack
      of points and their associated mass flux values.
      TODO: I imagine that we will get depth data from the plume model,
      but for right now, we just evenly space the depths of the
      mass flux values.
    '''
    def __init__(self, lon, lat,
                 plume_data):
        self.mass_flux = np.array([d[1] for d in plume_data])
        num_points = self.mass_flux.size

        self.coords = np.zeros((num_points), dtype=world_point)
        self.coords[:]['long'] = lon
        self.coords[:]['lat'] = lat
        self.coords[:]['z'] = np.array([d[0] for d in plume_data])


class PlumeGenerator(object):
    '''
      Here we define the method for generating LEs from a 3D plume
      over a range of time.
    '''
    def __init__(self,
                 release_time, end_release_time, time_step_delta,
                 plume):
        self.release_time = release_time
        self.end_release_time = end_release_time
        self.time_step_delta = time_step_delta
        self.time_steps = int((end_release_time - release_time).total_seconds() / time_step_delta)

        self.plume = plume

        self.accum_mass = np.zeros((self.plume.mass_flux.size))

        # Here we just calculate a reasonable value for the mass
        # that is contained in a single LE.
        # This may not be a good assumption, as other things may be
        # determining the mass of an LE in the model.
        # But we can always change this after class initialization
        # if we need to.
        self.mass_of_an_le = self.plume.mass_flux.mean() * time_step_delta * 2

    def _mass_to_elems(self, mass):
        '''
          Calculate mass into an equivalent number of LEs and return them.
          - We do not count the fractional amounts.
        '''
        return mass.astype(long) / self.mass_of_an_le.astype(long)

    def _elems_to_mass(self, elems):
        '''
          Calculate LEs into equivalent amounts of mass and return them.
        '''
        return elems * self.mass_of_an_le

    def _xfer_mass_to_elems(self):
        '''
          Transfer mass into an equivalent number of LEs and return them.
        '''
        tmp_elems = self._mass_to_elems(self.accum_mass)
        self.accum_mass -= self._elems_to_mass(tmp_elems)
    
        return tmp_elems

    def __iter__(self):
        for step in range(self.time_steps):
            self.accum_mass += self.plume.mass_flux * self.time_step_delta
            curr_step_time = self.release_time + timedelta(seconds=self.time_step_delta * step)
            yield (curr_step_time,
                   zip(self.plume.coords, self._xfer_mass_to_elems()))




def get_plume_data():
    '''
      - We will represent the mass flux amounts in kg/s.
      - We will probably get these values from a running
        plume model, but for right now, we just return
        an array with some hardcoded values.
      - For now, we return data in the format [(depth, mass_flux), ...]
    '''
    plume_mass_flux = np.zeros((10))
    plume_mass_flux[:] = 5.  # background values.
    plume_mass_flux[3] = 15. # and now a few spikes.
    plume_mass_flux[5] = 20.
    plume_mass_flux[7] = 30.

    plume_depths = np.linspace(0, 200, plume_mass_flux.size)
    
    return zip(plume_depths, plume_mass_flux)


if __name__ == '__main__':
    release_time = datetime.now()
    end_release_time = release_time + timedelta(hours=24)
    time_step_delta = timedelta(hours=1).total_seconds()

    plume = Plume(lon=10., lat=20.,
                  plume_data=get_plume_data())
    plume_gen = PlumeGenerator(release_time=release_time,
                               end_release_time=end_release_time,
                               time_step_delta=time_step_delta,
                               plume=plume)

    # let's print out some facts about our plume
    print '''
Based on the mean plume mass flux value,
we will choose an LE with %s kg of oil
''' % (plume_gen.mass_of_an_le)

    # now lets iterate our plume generator
    print 'First, just the occurrence pattern for LE releases...'
    for step in plume_gen:
        print step[0], [r[1] for r in step[1]]

    print '\nNext, the full information...'
    for step in plume_gen:
        for r in step:
            print r  # each row should be a world_point and a number of LEs to create
        print



