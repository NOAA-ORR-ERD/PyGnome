"""
module for manipulating netcdf particle files

This is a test case for working with what hopefully will be a CF standard

"""  # Change the / operator to ensure true division throughout (Zelenke).


from datetime import datetime

import numpy as np

import netCDF4


class particle_trajectory:

    def __init__(self):

        # lots of defaults:

        self.Trajectory = []
        self.timesteps = []
        self.time_units = 'hours since 2010-01-01 00:00:00'
        self.attributes = {
            'Conventions': 'CF-1.6',
            'title': 'Sample data/file for particle trajectory format',
            'institution': 'NOAA Emergency Response Division',
            'source': 'Example Data',
            'history': 'Evolved with discussion on CF-metadata listserve',
            'references': '',
            'comment': 'Some simple test data',
            'CF:featureType': 'particle_trajectory',
            }

#    dimensions:
#   record = UNLIMITED ; // (15768 currently)
#   time = 145 ;
# variables:
#
#       int number_records_per_timestep(time) ;
#       number_records_per_timestep:units = "1" ;
#       number_records_per_timestep:long_name = "number spillets per current timestep" ;
#       number_records_per_timestep:CF\:ragged_row_count = "record" ;
#   double time(record) ;
#       time:long_name = "forecast time after simulation start" ;
#       time:standard_name = "forecast_reference_time" ;
#       time:units = "days since 2010-05-25" ;
#       time:calendar = "gregorian" ;
#   float latitude(record) ;
#       latitude:long_name = "contaminant slick latitude" ;
#       latitude:standard_name = "latitude" ;
#       latitude:units = "degrees_north" ;
#   float longitude(record) ;

    def write(self, filename):
        nc = netCDF4.Dataset(filename, 'w', format='NETCDF3_CLASSIC')

        # Global Attributes

        for (attr, val) in list(self.attributes.items()):
            nc.setncattr(attr, val)
            setattr(nc, 'creation_date', datetime.now().isoformat())

        # add Dimensions

        print('num timesteps:', len(self.Trajectory))
        nc.createDimension('time', len(self.Trajectory))
        nc.createDimension('data', None)

        # create variables

        print('creating variables')

        # # fixme: should be able to create these from Trajectory dtype.

        Variables = {
            'time': nc.createVariable('time', 'f4', ('time', )),
            'particle_count': nc.createVariable('particle_count', 'i4',
                    'time'),
            'longitude': nc.createVariable('longitude', 'f4', ('data',
                    )),
            'latitude': nc.createVariable('latitude', 'f4', ('data',
                    )),
            'depth': nc.createVariable('depth', 'f4', ('data', )),
            'mass': nc.createVariable('mass', 'f4', ('data', )),
            'flag': nc.createVariable('flag', 'i1', ('data', )),
            'id': nc.createVariable('id', 'i4', ('data', )),
            }

        # diameter  = nc.createVariable('diameter', 'f4',('time','particle'))
        # rise_velocity  = nc.createVariable('rise_velocity', 'f4',('time','particle'))
        # vol = nc.createVariable('vol', 'f4',('time','particle'))
        # density  = nc.createVariable('density', 'f4',('time','particle'))
        # release_time  = nc.createVariable('release_time', 'f4',('time','particle'))

        # add data

        print('adding data')

        # #fixme -- should be able to add on by time -- how?

        for field in self.Trajectory[0].dtype.names:
            data = []
            for t in range(len(self.Trajectory)):
                data.extend(self.Trajectory[t][field])
                Variables['particle_count'][t] = len(self.Trajectory[t])
            (Variables[field])[:] = data

        # time

        (Variables['time'])[:] = netCDF4.date2num(self.timesteps,
                self.time_units)

        # # add attributes:

        Variable_attributes = {}
        Variable_attributes['time'] = {'long_name': 'Time',
                'standard_name': 'time', 'units': self.time_units}

        Variable_attributes['particle_count'] = {'units': '1',
                'long_name': 'number particles in given timestep',
                'CF:ragged_row_count': 'data'}

        Variable_attributes['lon'] = \
            {'long_name': 'Longitude of the particle'}

#            lon.standard_name = "longitude_particle"
#            lon.Fill_Value = -99999.
#            lon.missing_value = -99999.
#            lon.units = "degrees"

        Variable_attributes['lat'] = \
            {'long_name': 'Latitude of the particle'}

#            lat.long_name = "Latitude of the particle"
#            lat.standard_name = "latitude_particle"
#            lat.Fill_Value = -99999.
#            lat.missing_value = -99999.
#            lat.units = "degrees"

#            depth.long_name = "Depth of the particles"
#            depth.standard_name = "depth_particle"
#            depth.Fill_Value = -99999.
#            depth.missing_value = -99999.
#            depth.units = "meter"

        for (var, attrs) in list(Variable_attributes.items()):
            for (att, val) in list(attrs.items()):
                Variables[var].setncattr(att, val)
        nc.close()


# DSM: Wrapping of particle data generated by GNOME simulation

class nc_particle_file:

    """
    class to wrap a NetCDF particle file
    """

    def __init__(self, nc):

        self.nc = nc

        time = nc.variables['time']
        units = time.getncattr('units')
        self.times = netCDF4.num2date(time, units)
        self.time_units = units

        # Defined mass in the same way as done above for time (Zelenke).

        mass = nc.variables['mass']
        units_mass = mass.getncattr('units')
        self.mass = mass
        self.mass_units = units_mass

        # print nc.variables

        self.particle_count = nc.variables['particle_count']

        # build the index:

        self.data_index = np.zeros((len(self.times) + 1, ),
                                   dtype=np.int32)
        self.data_index[1:] = np.cumsum(self.particle_count)

        # print self.times
        # print self.particle_count
        # print self.data_index

    def get_all_timesteps(self, variables=['latitude', 'longitude']):
        """
         returns the requested variables data from a given timestep as a
         dictionary keyed by the variable names
         """

        data = {}
        for var in variables:
            data[var] = (self.nc.variables[var])[:]
        return data

    def get_timestep(self, timestep, variables=['latitude', 'longitude'
                     ]):
        """
        returns the requested variables data from a given timestep as a
        dictionary keyed by the variable names
        """

        data = {}
        for var in variables:
            ind1 = self.data_index[timestep]
            ind2 = self.data_index[timestep + 1]
            data[var] = (self.nc.variables[var])[ind1:ind2]
        return data

    def get_individual_trajectory(self, particle_id, vars=['latitude',
                                  'longitude']):
        """
        returns the requested variables from trajectory of an individual particle
        
        note: this is very inefficient -- it has to read the entire file to get it.
        """

        indexes = np.where(self.nc.variables['id'] == particle_id)


        # print self.nc.variables['id'][:]
        # print indexes

